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

def read_dates(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

train_dates = read_dates('splits/train.txt')
val_dates = read_dates('splits/val.txt')
test_dates = read_dates('splits/test.txt')


model = GAT(nfeat=64, nhid=64, nclass=2, dropout=0.3, alpha=0.2, nheads=8, stock_num=stock_num).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()

BATCH_SIZE = 8  # number of dates per batch
num_epochs = 30

best_val_f1 = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0
    all_preds = []
    all_targets = []

    for i in range(0, len(train_dates), BATCH_SIZE):
        batch_dates = train_dates[i:i + BATCH_SIZE]

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

    acc = total_correct / total_labels
    f1 = f1_score(all_targets, all_preds, average='micro')

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for date in val_dates:
            price_input = torch.tensor(np.load(f'{price_path}/{date}.npy'), dtype=torch.float32).to(device)
            text_input = torch.tensor(np.load(f'{text_path}/{date}.npy'), dtype=torch.float32).to(device)
            labels = torch.tensor(np.load(f'{label_path}/{date}.npy'), dtype=torch.long).to(device)

            out = model(text_input, price_input, adj)
            preds = torch.argmax(out, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_targets, val_preds, average='micro')

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model

    print(f"Epoch {epoch:3d} | Train Loss: {total_loss / len(train_dates):.4f} | Train F1: {f1:.4f} | Val F1: {val_f1:.4f}")
print("✅ Training complete.")
