import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
model.eval()

raw_data_path = "raw_tweets/"  # You need to adjust this
output_path = "train_text_finbert/"  # Where new embeddings are saved

os.makedirs(output_path, exist_ok=True)

def embed_text(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # [CLS] token average

for fname in os.listdir(raw_data_path):
    if fname.endswith(".txt"):
        with open(os.path.join(raw_data_path, fname), "r") as f:
            tweets = f.readlines()
        embeddings = []
        for tweet in tweets:
            embeddings.append(embed_text(tweet.strip()))
        embeddings = np.stack(embeddings)
        np.save(os.path.join(output_path, fname.replace(".txt", ".npy")), embeddings)
