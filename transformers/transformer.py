import torch
import torch.nn as nn
import torch.nn.functional as F

_fn = './input.txt'

token_to_idx = {}
idx_to_token = {}

context_size = 20
batch_size = 32
n_embd = 32
learning_rate = 1e-04
iters = 100

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
print(f"shapes: {X.shape} {y.shape}")
def get_batch():
    _perm = torch.randperm(X.shape[0])
    _data = {
        'X': X[_perm][:batch_size, :],
        'y': y[_perm][:batch_size, :]
    }
    return _data

class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_size, n_embd)
        self.linear = nn.Linear(n_embd*context_size, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(context_size))
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