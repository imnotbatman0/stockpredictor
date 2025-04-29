
# train.py (revised for full market batch per date)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef
from utils import load_data, accuracy
from model import GAT

adj = load_data().cuda()
stock_num = adj.shape[0]

price_path = 'price/'
text_path = 'text/'
label_path = 'label/'

dates = sorted([f.replace('.npy', '') for f in os.listdir(price_path)])

model = GAT(nfeat=64, nhid=64, nclass=2, dropout=0.3, alpha=0.2, nheads=8, stock_num=stock_num).cuda()
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, 201):
    model.train()
    date = np.random.choice(dates)

    price_input = torch.tensor(np.load(f'{price_path}/{date}.npy'), dtype=torch.float32).cuda()
    text_input = torch.tensor(np.load(f'{text_path}/{date}.npy'), dtype=torch.float32).cuda()
    labels = torch.tensor(np.load(f'{label_path}/{date}.npy'), dtype=torch.long).cuda()

    optimizer.zero_grad()
    out = model(text_input, price_input, adj)
    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = torch.argmax(out, dim=1)
        acc = accuracy(out, labels).item()
        f1 = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='micro')
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
