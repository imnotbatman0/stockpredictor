# preprocess_tweets.py

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
model.eval()

MAX_TWEETS = 10
EMBED_DIM = 768
PROJECT_DIM = 64

projection = torch.nn.Linear(EMBED_DIM, PROJECT_DIM).to(device)

def embed_tweet(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = projection(pooled)
        return projected.squeeze(0).cpu().numpy()

def preprocess_tweets(tweet_folder, price_dates):
    tweet_data = {}
    for ticker in os.listdir(tweet_folder):
        ticker_path = os.path.join(tweet_folder, ticker)
        if os.path.isdir(ticker_path):
            ticker_tweets = {}
            for date in price_dates.get(ticker, []):
                date_path = os.path.join(ticker_path, date)
                tweets = []
                if os.path.exists(date_path):
                    with open(date_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for line in lines:
                        tweets.append(embed_tweet(line.strip()))

                # Pad or trim
                if len(tweets) == 0:
                    tweets = [np.zeros(PROJECT_DIM) for _ in range(MAX_TWEETS)]
                elif len(tweets) < MAX_TWEETS:
                    tweets += [np.zeros(PROJECT_DIM) for _ in range(MAX_TWEETS - len(tweets))]
                else:
                    tweets = tweets[:MAX_TWEETS]

                ticker_tweets[date] = np.stack(tweets)

            tweet_data[ticker] = ticker_tweets
            print(ticker)

    return tweet_data

