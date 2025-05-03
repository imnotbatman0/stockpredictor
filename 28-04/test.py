
import torch
import numpy as np
from model import GAT
from utils import load_data
from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dates = [line.strip() for line in open('splits/test.txt')]
adj = load_data().to(device)
stock_num = adj.shape[0]

model = GAT(nfeat=64, nhid=64, nclass=2, dropout=0.3, alpha=0.2, nheads=8, stock_num=stock_num).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

price_path = 'price/'
text_path = 'text/'
label_path = 'label/'

all_preds, all_labels = [], []
with torch.no_grad():
    for date in test_dates:
        price_input = torch.tensor(np.load(f'{price_path}/{date}.npy'), dtype=torch.float32).to(device)
        text_input = torch.tensor(np.load(f'{text_path}/{date}.npy'), dtype=torch.float32).to(device)
        labels = torch.tensor(np.load(f'{label_path}/{date}.npy'), dtype=torch.long).to(device)

        out = model(text_input, price_input, adj)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

f1 = f1_score(all_labels, all_preds, average='micro')
print(f"✅ Test F1 Score: {f1:.4f}")
