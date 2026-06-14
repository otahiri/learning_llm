# vocab = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
# merges = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/merges.txt"
# tokenizer = "/home/otahiri-/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
# special_chars = {' ' : 'Ġ', "\n" : 'Ċ', '\t' : 'ĉ'}


vocab_cap = 1000
new_token = 256
merges: dict = dict()


def decode(token: int, translation: dict) -> str:
    if token < 256:
        return chr(token)
    else:
        return translation[token]


def encode(text: str):

    tokens = list(text.encode('utf-8'))

    while True:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        pair_to_merge = None
        lowest_id = float('inf')
        for p in pairs:
            if p in merges and merges[p] < lowest_id:
                pair_to_merge = p
                lowest_id = merges[p]

        if not pair_to_merge:
            break
        new_tokens: list = []
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair_to_merge:
                tokens[i:i+2] = [lowest_id]
            else:
                i += 1
    return tokens


def update_vocab(encoded: list, pairs: list, vocab: set, translation: dict):
    global new_token
    pair, count = pairs[0]
    if count > 1 and len(vocab) < vocab_cap:
        vocab.add("".join([decode(x, translation) for x in pair]))
        i = 0
        translation[new_token] = "".join([decode(x, translation) for x in pair])
        merges[pair] = new_token
        while i < len(encoded) - 1:
            if (encoded[i], encoded[i + 1]) == pair:
                encoded[i: i + 2] = [new_token]
            else:
                i += 1
        new_token += 1


def count_pairs(text: list) -> list:

    count: dict = dict()

    i = 0
    while i < len(text) - 1:
        count[(text[i], text[i + 1])] = count.get((text[i], text[i + 1]), 0) + 1
        i += 1
    return sorted(count.items(), key=lambda x: x[1], reverse=True)


def main() -> None:
    text = "hellor world! howhow howare howhhhhhhow you doing or, or or or or"

    vocab: set = set(text)

    encoded = list(text.encode("utf-8"))

    translation: dict = {list(a.encode("utf-8"))[0]: a for a in vocab}
    while True:
        pairs = count_pairs(encoded)
        if pairs and pairs[0][1] < 2:
            break
        update_vocab(encoded, pairs, vocab, translation)
    print(translation)
    print(encode(text))
    print("".join([decode(x, translation) for x in encode(text)]))


if __name__ == "__main__":
    main()
