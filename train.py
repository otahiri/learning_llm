from os import login_tty
from urllib import request
from typing import Any
import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
text = ""
chars = []

url = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt"
file_name = "input.txt"
request.urlretrieve(url, file_name)

with open(file_name, "r") as f:
    text = f.read()
    chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

encode = lambda s: [stoi.get(c) for c in s]
decode = lambda l: "".join(itos[i] for i in l)

res = encode(text)
data = torch.tensor(res, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
torch.manual_seed(42)


class BigramLLM(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i+1 + block_size] for i in ix])
    return x, y


def main():
    xb, yb = get_batch('train')
    m = BigramLLM(vocab_size)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for steps in range(60000):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss.item())

    print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_token=400)[0].tolist()))



if __name__ == "__main__":
    main() 
