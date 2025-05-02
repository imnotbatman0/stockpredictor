# train.py — trains over sequential full epochs (no random sampling)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, matthews_corrcoef
from utils import load_data, accuracy
from model import GAT

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load graph and model
adj = load_data().to(device)
stock_num = adj.shape[0]

price_path = 'price/'
text_path = 'text/'
label_path = 'label/'
dates = sorted([f.replace('.npy', '') for f in os.listdir(price_path)])

model = GAT(nfeat=64, nhid=64, nclass=2, dropout=0.3, alpha=0.2, nheads=8, stock_num=stock_num).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 8  # number of dates per batch
num_epochs = 500

# Training loop (sweep sequentially through all dates per epoch)
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0
    all_preds = []
    all_targets = []

    for i in range(0, len(dates), BATCH_SIZE):
        batch_dates = dates[i:i + BATCH_SIZE]

        for date in batch_dates:
            price_input = torch.tensor(np.load(f'{price_path}/{date}.npy'), dtype=torch.float32).to(device)
            text_input = torch.tensor(np.load(f'{text_path}/{date}.npy'), dtype=torch.float32).to(device)
            labels = torch.tensor(np.load(f'{label_path}/{date}.npy'), dtype=torch.long).to(device)

            optimizer.zero_grad()
            out = model(text_input, price_input, adj)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            total_correct += (preds == labels).sum().item()
            total_labels += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = total_correct / total_labels if total_labels > 0 else 0.0
    f1 = f1_score(all_targets, all_preds, average='micro') if total_labels > 0 else 0.0

    if epoch % 1 == 0:
        print(f"Epoch {epoch:3d} | Avg Loss: {total_loss / len(dates):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

print("✅ Training complete.")
