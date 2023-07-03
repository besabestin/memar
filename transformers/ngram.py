import torch
import torch.nn.functional as F
from torch import nn, optim


"""
just a simple n-gram language model
currently a tri-gram language model
no transformers no nothin
"""

CONTEXT_SIZE = 2
EMBEDDING_DIM = 100
batch_size = 32
epochs = 1000

text = ""

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text[:5000]
text = text.split()

trigram = [((text[i], text[i + 1]), text[i + 2])
            for i in range(len(text) - 2)]

vocb = set(text)

word_to_idx = {word: i for i,word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim) -> None:
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)
    
    def forward(self, x, train=True):
        emb = self.embedding(x)
        
        if train:
            emb = emb.view(batch_size, -1)
        else:
            emb = emb.view(1, -1)
        
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        
        log_prob = F.log_softmax(out, dim=1)
        
        return log_prob

    def generate(self, x, max_tokens=100):
        for _ in range(max_tokens):
            out = self(x[-2:], train=False)
            _, predict_label = torch.max(out, 1)
            x = torch.cat((x, torch.tensor([predict_label], dtype=torch.long)), dim=-1)
        str = ""
        for idx in x:
            str = f"{str} {idx_to_word[idx.item()]}"
        print(str)


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.NLLLoss()
optimizer = optim.Adam(ngrammodel.parameters(), lr=1e-03)


def get_batch():
    ix = torch.randint(len(trigram), (batch_size,))
    
    xtensors = []
    ytensors = []
    for i in ix:
        word, label = trigram[i]
        xtensors.append(torch.tensor([word_to_idx[val] for val in word], dtype=torch.long))
        ytensors.append(torch.tensor(word_to_idx[label], dtype=torch.long))

    x = torch.stack(xtensors)
    y = torch.stack(ytensors)
    return x,y

X, yy = get_batch()
print(f"the shapes: {X.shape} {yy.shape}")

for epoch in range(epochs):
    X, y = get_batch()
    
    out = ngrammodel(X)
    loss = criterion(out, y)
    epochloss = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 99 == 0:
        print(f'epoch: {epoch}')
        print(f'Loss: {epochloss}')


# to generate
with torch.no_grad():
    word = ['so', 'they']
    word = torch.tensor([word_to_idx[val] for val in word], dtype=torch.long)
    ngrammodel.generate(word)