vocab_file = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
merge_file = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/merges.txt"
# tokenizer = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
special_chars = {" ": "Ġ", "\n": "Ċ", "\t": "ĉ"}

import json

new_token = 256
merges: dict = dict()


def decode(token: int, translation: dict) -> str:
    return translation.get(token, "")


def encode(text: str, vocab: dict):

    tokens = [vocab.get(c, 0) for c in text]
    while True:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        pair_to_merge = None
        lowest_rank = float('inf')
        merge_id = None
        for p in pairs:
            if p in merges:
                target_id, rank = merges[p]
                if rank < lowest_rank:
                    lowest_rank = rank
                    pair_to_merge = p
                    merge_id = target_id

        if not pair_to_merge:
            break
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair_to_merge:
                tokens[i: i + 2] = [merge_id]
            else:
                i += 1
    return tokens


def main() -> None:
    global new_token

    vocab: dict = dict()
    with open(vocab_file, "r") as f:
        vocab = dict(json.load(f))
    text = "hello world, my name is oussama"
    with open(merge_file, "r") as f:
        lines = f.read().splitlines()
        for rank, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            parts = line.split(' ')
            if len(parts) == 2:
                p1, p2 = parts
                merge_str = p1 + p2
                if p1 in vocab and p2 in vocab and merge_str in merges:
                    merges[(vocab[p1], vocab[p2])] = (vocab[merge_str], rank)
            new_token += 1

    for key, value in special_chars.items():
        text = text.replace(key, value)

    str_to_id = {v: k for k, v in vocab.items()}
    encoded = encode(text, vocab)
    print(encoded)
    text = "".join([decode(x, str_to_id) for x in encoded])

    for key, value in special_chars.items():
        text = text.replace(value, key)
    print(text)


if __name__ == "__main__":
    main()
