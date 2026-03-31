from transformerlm.tokenizer.pretokenize import gpt2_bytes_to_unicode
from transformerlm.tokenizer.io import (
    find_chunk_boundaries,
    process_chunk_text,
)
import json
from collections import Counter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from typing import Optional
from leash.logger import Logger


def train_bpe(args, *, logger: Optional[Logger] = None):
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

     # - initialize vocab with <|endoftext|> and gpt2 encoder
    vocab_list = [gpt2_byte_encoder[k] for k in gpt2_byte_encoder.keys()]
    initial_vocab = {k:i for i, k in enumerate(vocab_list)}

    # - open corpus, remove special token and pretokenize on chunks
    num_processes = 1
    pretoken_frequency = Counter()
    with open(args.input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f,
                                           num_processes,
                                           "<|endoftext|>".encode("utf-8"))
        
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    partial(process_chunk_text, input_path=args.input_path, special_tokens=args.special_tokens),
                    zip(boundaries[:-1], boundaries[1:])
                )
            )
        for partial_pretoken_frequency in results:
            pretoken_frequency.update(partial_pretoken_frequency)

    pretoken_frequency_dict = dict(pretoken_frequency)
    pretoken_frequency_tuple = {
        tuple(gpt2_byte_encoder[b] for b in word.encode('utf-8')): freq
        for word, freq in pretoken_frequency_dict.items()
    }

    # - compute bpe merges until vocab size is reached
    merges = []
    vocab = initial_vocab
    iter = 0
    iter_max = args.vocab_size - len(list(vocab.keys()))
    last_log = time.time()
    while iter < iter_max:
        # compute pair frequency
        pair_frequency = {}
        for freq_tuple, freq_value in pretoken_frequency_tuple.items():
            for i in range(len(freq_tuple)-1):
                pair = (freq_tuple[i], freq_tuple[i+1])
                if pair in pair_frequency: 
                    pair_frequency[pair] = pair_frequency[pair] + freq_value
                elif not pair in pair_frequency:
                    pair_frequency[pair] = freq_value
        
        # get biggest freq and update merges
        max_value = max(pair_frequency.values())
        max_pair_freq = [k for k, v in pair_frequency.items() if v == max_value]
        if len(max_pair_freq)>1:
            best_key = max(max_pair_freq, key=lambda p: ("".join([chr(gpt2_byte_decoder[c]) for c in p[0]]), "".join([chr(gpt2_byte_decoder[c]) for c in p[1]])))
        else:
            best_key = max_pair_freq[0]

        merges.append(best_key[0]+" "+best_key[1])

        # update pretoken frequency tuples with new merges
        new_merge = ''.join(best_key)
        new_pretoken_frequency_tuple = defaultdict(int)

        for key, value in pretoken_frequency_tuple.items():
            new_key = []
            i = 0
            while i < len(key):
                # Check for the merge pair
                if key[i:i+2] == best_key:
                    new_key.append(new_merge)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_pretoken_frequency_tuple[tuple(new_key)] += value

        pretoken_frequency_tuple = dict(new_pretoken_frequency_tuple)

        iter += 1
        # periodic logging
        if logger is not None and iter % 100 == 0:
            now = time.time()
            dt = now - last_log
            last_log = now
            logger.log(
                {
                    "phase": "tokenizer",
                    "metrics.merges_applied": int(iter),
                    "metrics.vocab_size": int(len(vocab) + len(merges)),
                    "metrics.merges_per_sec": float(100.0 / dt) if dt > 0 else float("inf"),
                },
                step=iter,
            )

    # - return vocab w/ additional merges and merges as bytes
    initial_vocab_size = len(list(vocab.keys()))
    for i, merge in enumerate(merges):
        merge_token = "".join(merge.split())
        vocab[merge_token] = initial_vocab_size + i
    for i, special_token in enumerate(args.special_tokens):
        vocab[special_token] = len(vocab)+i

    with open(args.merges_path, "w", encoding="utf-8") as f:
        for token in merges:
            f.write(token + "\n")

    with open(args.vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    # Log final artifacts
    if logger is not None:
        try:
            logger.log(
                {
                    "phase": "tokenizer",
                    "event": "finalize",
                    "metrics.merges_applied": int(iter),
                    "metrics.vocab_size": int(len(vocab)),
                },
                step=iter,
            )
            logger.log_artifact(str(args.merges_path), name=str(args.merges_path), type_="tokenizer")
            logger.log_artifact(str(args.vocab_path), name=str(args.vocab_path), type_="tokenizer")
        except Exception:
            pass
