from pathlib import Path
from urllib import request
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 256
head_count = 6
eval_iters = 200
cycles = 6
max_iters = 5000
eval_interval = 500
n_embd = 384
learning_rate = 3e-4
dropout = 0.2
text = ""
device = "cuda" if torch.cuda.is_available() else "cpu"
chars = []

file_name = "input.txt"
url = "https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt"

if not Path(file_name).exists():
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
torch.manual_seed(1337)


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        return out


class MultiHead(nn.Module):
    def __init__(self, head_count, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffwrd = FeedForward()
        self.multi_head = MultiHead(head_count, n_embd // head_count)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head(self.layer_norm1(x))
        x = x + self.ffwrd(self.layer_norm2(x))
        return x


class BigramLLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position = nn.Embedding(block_size, n_embd)
        self.block = nn.Sequential(*[Block() for _ in range(cycles)])
        self.ln1 = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.block(x)
        x = self.ln1(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch("train")
m = BigramLLM()
m = m.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


def main():
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device)
    for steps in range(max_iters):
        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {steps}: val loss {losses['val']:.4f}, train loss {losses['train']:.4f}"
            )

        xb, yb = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = m(xb, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.save(m.state_dict(), "shakespeare_model.pt")
    print(
        decode(
            m.generate(
                torch.zeros((1, 1), dtype=torch.long, device=device), max_token=400
            )[0].tolist()
        )
    )


if __name__ == "__main__":
    main()
