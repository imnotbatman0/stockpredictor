# train.py — trains over sequential full epochs, saves model and metrics

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

BATCH_SIZE = 8
num_epochs = 5000
val_split = int(len(dates) * 0.9)
val_dates = dates[val_split:]

# Lists to store metrics
train_accs, train_f1s, val_accs, val_f1s = [], [], [], []

# Training loop
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

    train_acc = total_correct / total_labels if total_labels > 0 else 0.0
    train_f1 = f1_score(all_targets, all_preds, average='micro') if total_labels > 0 else 0.0

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for date in val_dates:
            price_input = torch.tensor(np.load(f'{price_path}/{date}.npy'), dtype=torch.float32).to(device)
            text_input = torch.tensor(np.load(f'{text_path}/{date}.npy'), dtype=torch.float32).to(device)
            labels = torch.tensor(np.load(f'{label_path}/{date}.npy'), dtype=torch.long).to(device)

            out = model(text_input, price_input, adj)
            preds = torch.argmax(out, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    val_acc = accuracy(val_preds, val_targets)
    val_f1 = f1_score(val_targets, val_preds, average='micro')

    # Store metrics
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch:3d} | Train Loss: {total_loss / len(dates):.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

# Save model
torch.save(model.state_dict(), 'gat_model.pth')
print("✅ Model saved to gat_model.pth")

# Save metrics
np.save('metrics.npy', {
    'train_acc': train_accs,
    'train_f1': train_f1s,
    'val_acc': val_accs,
    'val_f1': val_f1s
})
print("📊 Metrics saved to metrics.npy")

# Plot metrics
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, val_accs, label='Val Acc')
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Training & Validation Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('metrics_plot.png')
plt.close()
print("📈 Plot saved to metrics_plot.png")
