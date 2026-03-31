import torch

from leash.data.streaming import HFTokenIteratorFactory, StreamingBatcher


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 256 for ch in text]


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, buffer_size: int, seed: int):
        return self

    def shard(self, num_shards: int, index: int):
        rows = [row for idx, row in enumerate(self.rows) if idx % num_shards == index]
        ds = FakeDataset(rows)
        ds.num_shards = num_shards
        return ds


class TestHFTokenIteratorFactory(HFTokenIteratorFactory):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self._dataset = dataset

    def _load_dataset(self):
        return self._dataset

    def _apply_shuffle(self, dataset):
        return dataset


def test_hf_token_iterator_appends_newline_and_eot():
    dataset = FakeDataset([{"text": "hi"}])
    tokenizer = DummyTokenizer()
    factory = TestHFTokenIteratorFactory(
        dataset,
        dataset_name="fake",
        dataset_config=None,
        split="train",
        text_field="text",
        tokenizer=tokenizer,
        context_length=8,
        eot_token_id=99,
        pad_newline=True,
        world_size=1,
        rank=0,
    )
    tokens = next(iter(factory()))
    assert tokens[:-1] == tokenizer.encode("hi\n")
    assert tokens[-1] == 99


def test_hf_token_iterator_sharding():
    dataset = FakeDataset([{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}])
    tokenizer = DummyTokenizer()
    factory = TestHFTokenIteratorFactory(
        dataset,
        dataset_name="fake",
        dataset_config=None,
        split="train",
        text_field="text",
        tokenizer=tokenizer,
        context_length=8,
        eot_token_id=7,
        pad_newline=False,
        world_size=2,
        rank=1,
    )
    it = iter(factory())
    first = next(it)
    second = next(it)
    assert first[:-1] == tokenizer.encode("b")
    assert second[:-1] == tokenizer.encode("d")


def test_streaming_batcher_packs_from_iterator(device):
    dataset = FakeDataset([{"text": "ab"}, {"text": "cd"}])
    tokenizer = DummyTokenizer()
    factory = TestHFTokenIteratorFactory(
        dataset,
        dataset_name="fake",
        dataset_config=None,
        split="train",
        text_field="text",
        tokenizer=tokenizer,
        context_length=8,
        eot_token_id=7,
        pad_newline=False,
        world_size=1,
        rank=0,
    )
    batcher = StreamingBatcher(factory, device=str(device))
    batch = batcher.draw(batch_size=1, context_length=4)
    expected = tokenizer.encode("ab") + [7] + tokenizer.encode("c")
    assert batch.shape == (1, 4)
    assert batch[0].tolist() == expected
