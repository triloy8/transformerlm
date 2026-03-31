from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterator, Optional

import numpy as np
import torch

from leash.logger import Logger


_INDEX_HEADER = b"MMIDIDX\x00\x00"


@dataclass
class _Index:
    dtype: type
    sequence_lengths: np.ndarray
    sequence_pointers: np.ndarray
    document_indices: np.ndarray


def _read_index(idx_path: str) -> _Index:
    with open(idx_path, "rb") as stream:
        header = stream.read(9)
        if header != _INDEX_HEADER:
            raise ValueError(f"bad header, cannot read: {idx_path}")
        version = struct.unpack("<Q", stream.read(8))[0]
        if version != 1:
            raise ValueError(f"bad version, cannot read: {idx_path}")
        code = struct.unpack("<B", stream.read(1))[0]
        dtype = _dtype_from_code(code)
        sequence_count = struct.unpack("<Q", stream.read(8))[0]
        document_count = struct.unpack("<Q", stream.read(8))[0]
        offset = stream.tell()

    buffer_mmap = np.memmap(idx_path, mode="r", order="C")
    buffer = memoryview(buffer_mmap)
    sequence_lengths = np.frombuffer(buffer, dtype=np.int32, count=sequence_count, offset=offset)
    sequence_pointers = np.frombuffer(
        buffer,
        dtype=np.int64,
        count=sequence_count,
        offset=offset + sequence_lengths.nbytes,
    )
    document_indices = np.frombuffer(
        buffer,
        dtype=np.int64,
        count=document_count,
        offset=offset + sequence_lengths.nbytes + sequence_pointers.nbytes,
    )
    return _Index(dtype=dtype, sequence_lengths=sequence_lengths, sequence_pointers=sequence_pointers, document_indices=document_indices)


def _dtype_from_code(code: int) -> type:
    mapping = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float64,
        7: np.float32,
        8: np.uint16,
    }
    if code not in mapping:
        raise ValueError(f"unsupported dtype code: {code}")
    return mapping[code]


class MegatronIndexedDataset:
    def __init__(self, path_prefix: str) -> None:
        self.path_prefix = path_prefix
        idx_path = f"{path_prefix}.idx"
        bin_path = f"{path_prefix}.bin"
        self.index = _read_index(idx_path)
        self._bin_mmap = np.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_mmap)

    def __len__(self) -> int:
        return int(self.index.sequence_lengths.shape[0])

    def get_sequence(self, idx: int) -> np.ndarray:
        pointer = int(self.index.sequence_pointers[idx])
        length = int(self.index.sequence_lengths[idx])
        return np.frombuffer(self._bin_buffer, dtype=self.index.dtype, count=length, offset=pointer)

    def __del__(self) -> None:
        if hasattr(self, "_bin_mmap"):
            self._bin_mmap._mmap.close()


class MegatronPackedBatcher:
    """Packed-stream batcher that draws from a Megatron indexed dataset."""

    def __init__(
        self,
        path_prefix: str,
        *,
        device: str | torch.device,
        logger: Optional[Logger] = None,
    ) -> None:
        self.dataset = MegatronIndexedDataset(path_prefix)
        self.device = torch.device(device)
        self._cursor = 0
        self._buffer: list[int] = []
        self._logger = logger
        self._epoch = 0

    def _extend_buffer(self) -> None:
        while True:
            if self._cursor >= len(self.dataset):
                self._cursor = 0
                self._epoch += 1
                if self._logger is not None:
                    self._logger.log(
                        {
                            "metrics.data/megatron_epoch": int(self._epoch),
                            "metrics.data/megatron_reset": 1,
                        }
                    )
            seq = self.dataset.get_sequence(self._cursor)
            self._cursor += 1
            if seq.size == 0:
                continue
            self._buffer.extend(int(x) for x in seq.tolist())
            return

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        sequences: list[list[int]] = []
        for _ in range(batch_size):
            while len(self._buffer) < context_length:
                self._extend_buffer()
            seq = self._buffer[:context_length]
            self._buffer = self._buffer[context_length:]
            sequences.append(seq)
        clean_targets = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return clean_targets

