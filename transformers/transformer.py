import torch
import torch.nn as nn
import torch.nn.functional as F

_fn = './input.txt'

token_to_idx = {}
idx_to_token = {}

context_size = 64
batch_size = 32
n_embd = 32*6
learning_rate = 1e-04
iters = 10000
dropout = 0.1
nhead = 6
ndecoders = 6

def encode(tokens):
    return [token_to_idx[token] for token in tokens]

def decode(idcs):
    return [idx_to_token[idx] for idx in idcs]

with open(_fn, 'r') as f:
    fulltext = f.read().replace('\n', ' ').split()
    vocab = set(fulltext)
    vocab_size = len(vocab)
    token_to_idx = {token: i for i, token in enumerate(vocab)}
    idx_to_token = {token_to_idx[token]: token for token in vocab}
    X = [torch.tensor(encode(fulltext[i:i+context_size]), dtype=torch.long) for i in range(len(fulltext) - context_size - 1)]
    y = [torch.tensor(encode([fulltext[i+context_size]]), dtype=torch.long) for i in range(len(fulltext) - context_size - 1)]
    X = torch.stack(X)
    y = torch.stack(y)

def get_batch():
    _perm = torch.randperm(X.shape[0])
    _data = {
        'X': X[_perm][:batch_size, :],
        'y': y[_perm][:batch_size, :]
    }
    return _data


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        B, T, C = x.shape
        kw = self.k(x)
        qw = self.q(x)
        vw = self.v(x)
        wei = kw@qw.transpose(-2, -1)*C**-0.5
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei@vw


class MultiheadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_size = n_embd // nhead
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nhead)])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out


class Feedforward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.ffwd(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sahead = MultiheadAttention()
        self.ffwd = Feedforward()
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sahead(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_size, n_embd)
        self.decoder = nn.Sequential(*[DecoderLayer() for _ in range(ndecoders)])
        self.linear = nn.Linear(n_embd*context_size, vocab_size)
        self.layernorm = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(context_size))
        x = self.layernorm(self.decoder(x))
        B, T, C = x.shape
        x = x.view(B, T*C)
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)
        return x


model = LanguageModel()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

for iter in range(iters):
    model.train()
    _batchdata = get_batch()
    out = model(_batchdata['X'])
    loss = loss_fn(out, _batchdata['y'].flatten())
    loss.backward()
    optim.step()
    optim.zero_grad()

    if iter%10 == 0:
        print(f"epoch: {iter}/{iters} loss: {loss}")