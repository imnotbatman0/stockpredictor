# embed_tweets.py — updated to handle JSON tweet files with text arrays

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MAX_TWEETS = 10
EMBED_DIM = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
model.eval()

def embed_tweet(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled.squeeze(0).cpu().numpy()

def embed_all_tweets(tweet_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    for ticker in os.listdir(tweet_root):
        ticker_path = os.path.join(tweet_root, ticker)
        out_path = os.path.join(output_root, ticker)
        os.makedirs(out_path, exist_ok=True)

        for fname in os.listdir(ticker_path):
            day_path = os.path.join(ticker_path, fname)
            if not os.path.isfile(day_path):
                continue

            with open(day_path, 'r', encoding='utf-8') as f:
                try:
                    tweets_raw = json.load(f)
                except json.JSONDecodeError:
                    continue

            # Sort by followers count descending
            tweets_raw.sort(key=lambda x: x.get('followers_count', 0), reverse=True)

            tweet_embeddings = []
            for entry in tweets_raw:
                if 'text' in entry and isinstance(entry['text'], list):
                    text = ' '.join(entry['text'])
                    tweet_embeddings.append(embed_tweet(text))
                if len(tweet_embeddings) >= MAX_TWEETS:
                    break

            # Pad if less than 10
            if len(tweet_embeddings) < MAX_TWEETS:
                tweet_embeddings += [np.zeros(EMBED_DIM) for _ in range(MAX_TWEETS - len(tweet_embeddings))]

            tweet_array = np.stack(tweet_embeddings)
            np.save(os.path.join(out_path, fname + ".npy"), tweet_array)

if __name__ == '__main__':
    embed_all_tweets("data/tweets", "tweet_embeds_768")
