import torch
import torch.nn.functional as F
from torch import nn, optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 100

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
    
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        #print(f"out shape: {out.shape}")
        log_prob = F.log_softmax(out, dim=1)
        #print(log_prob)
        return log_prob


ngrammodel = NgramModel(len(word_to_idx), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.NLLLoss()
optimizer = optim.Adam(ngrammodel.parameters(), lr=1e-03)

for epoch in range(100):
    #print(f'epoch: {epoch}')
    running_loss = 0
    for data in trigram:
        word, label = data
        word = torch.tensor([word_to_idx[val] for val in word], dtype=torch.long)
        label = torch.tensor(word_to_idx[label], dtype=torch.long)
        label = label.unsqueeze(0)

        #print(f"the label: {label}")
        # forward
        out = ngrammodel(word)
        loss = criterion(out, label)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 9 == 0:
        print(f'epoch: {epoch}')
        print('Loss: {:.6f}'.format(running_loss/len(word_to_idx)))


# to generate
word, label = trigram[50]
print(word)
word = torch.tensor([word_to_idx[val] for val in word], dtype=torch.long)
out = ngrammodel(word)
_, predict_label = torch.max(out, 1)
predict_word = idx_to_word[predict_label.item()]
print(f'real word: {label}, predict word: {predict_word}')