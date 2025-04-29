# Updated model.py with hierarchical tweet encoding

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, TransformerEncoderLayer, CrossAttentionLayer, GRUWithAttention, TemporalAttentionPooling

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, stock_num):
        super(GAT, self).__init__()
        self.stock_num = stock_num
        self.dropout = dropout

        # Price encoder
        self.price_transformers = nn.ModuleList([TransformerEncoderLayer(d_model=4, nhead=1) for _ in range(stock_num)])
        self.price_projectors = nn.ModuleList([nn.Linear(4, 64) for _ in range(stock_num)])

        # Intra-day tweet encoder (GRU + attention)
        self.intra_day_encoders = nn.ModuleList([GRUWithAttention(input_dim=64, hidden_dim=64) for _ in range(stock_num)])

        # Inter-day tweet encoder (Transformer + attention pooling)
        self.inter_day_transformers = nn.ModuleList([TransformerEncoderLayer(d_model=64, nhead=4) for _ in range(stock_num)])
        self.temporal_pooling = nn.ModuleList([TemporalAttentionPooling(input_dim=64) for _ in range(stock_num)])

        # Cross-Attention fusion
        self.cross_attention_layers = nn.ModuleList([CrossAttentionLayer(d_model=64, nhead=4) for _ in range(stock_num)])

        self.output_projectors = nn.ModuleList([nn.Linear(64, 2) for _ in range(stock_num)])

        # Graph layers
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, text_input, price_input, adj):
        li = []
        num_stocks = price_input.size(0)

        for i in range(num_stocks):
            # Price encoding
            price_seq = price_input[i]  # [5, 4]
            price_emb = self.price_transformers[i](price_seq.unsqueeze(0))
            price_emb = price_emb.mean(dim=1)
            price_emb = self.price_projectors[i](price_emb)

            # Intra-day tweet encoding (per day)
            daily_vecs = []
            for d in range(5):
                tweets = text_input[i, d]  # [10, 64]
                tweets = tweets.unsqueeze(0)  # [1, 10, 64]
                daily_vec = self.intra_day_encoders[i](tweets)  # [1, 64]
                daily_vecs.append(daily_vec)
            tweet_sequence = torch.stack(daily_vecs, dim=1).squeeze(0)  # [5, 64]

            # Inter-day encoding
            tweet_seq_encoded = self.inter_day_transformers[i](tweet_sequence.unsqueeze(0))  # [1, 5, 64]
            tweet_final = self.temporal_pooling[i](tweet_seq_encoded)  # [1, 64]

            # Cross attention fusion
            fused = self.cross_attention_layers[i](price_emb.unsqueeze(1), tweet_final.unsqueeze(1), tweet_final.unsqueeze(1))
            fused = fused.squeeze(1)
            fused = F.relu(fused)

            li.append(fused.unsqueeze(0))

        ft_vec = torch.cat(li, dim=0).view(num_stocks, -1)  # [num_stocks, 64]

        # Individual classifiers
        out_1 = torch.stack([self.output_projectors[i](ft_vec[i]) for i in range(num_stocks)], dim=0)

        # Graph layer
        x = F.dropout(ft_vec, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return x + out_1
