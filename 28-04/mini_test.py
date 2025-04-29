# mini_test.py

import torch
from model import GAT
from utils import load_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj = load_data().to(device)
    stock_num = adj.shape[0]

    model = GAT(nfeat=64, nhid=64, nclass=2, dropout=0.3, alpha=0.2, nheads=8, stock_num=stock_num).to(device)

    dummy_price = torch.randn(stock_num, 5, 4).to(device)  # [stocks, days, 4 features]
    dummy_text = torch.randn(stock_num, 5, 10, 64).to(device)  # [stocks, days, 10 tweets, 64 dims]

    with torch.no_grad():
        out = model(dummy_text, dummy_price, adj)
        print("Output shape:", out.shape)  # Should be [stocks, 2]
        print(out)