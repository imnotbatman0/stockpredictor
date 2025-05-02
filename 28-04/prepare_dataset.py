# prepare_dataset.py — Updated for 768-dim tweet embeddings

import os
import numpy as np
from preprocess_prices import preprocess_prices

price_folder = 'data/stocks/'
tweet_embed_folder = 'tweet_embeds_768/'

os.makedirs('price', exist_ok=True)
os.makedirs('text', exist_ok=True)
os.makedirs('label', exist_ok=True)

print("Processing prices...")
price_data = preprocess_prices(price_folder)  # {ticker: {'samples', 'labels', 'dates'}}

# Build master list of aligned dates
all_dates = sorted(list(set.intersection(*[set(price_data[t]['dates']) for t in price_data])))

print("Preparing dataset...")
for date_idx, date in enumerate(all_dates):
    price_tensor = []
    tweet_tensor = []
    label_tensor = []

    for ticker in price_data:
        try:
            idx = price_data[ticker]['dates'].index(date)
        except ValueError:
            continue

        # Load price vector
        price_tensor.append(price_data[ticker]['samples'][idx])  # [5, 4]
        label_tensor.append(price_data[ticker]['labels'][idx])

        # Load 5 tweet days
        tweet_days = []
        for j in range(5):
            prev_date = price_data[ticker]['dates'][idx - 4 + j]  # 5-day window
            embed_path = os.path.join(tweet_embed_folder, ticker, prev_date + ".npy")
            if os.path.exists(embed_path):
                tweet_days.append(np.load(embed_path))  # [10, 768]
            else:
                tweet_days.append(np.zeros((10, 768)))

        tweet_tensor.append(np.stack(tweet_days))  # [5, 10, 768]

    price_tensor = np.stack(price_tensor)   # [num_stocks, 5, 4]
    tweet_tensor = np.stack(tweet_tensor)   # [num_stocks, 5, 10, 768]
    label_tensor = np.array(label_tensor)   # [num_stocks]

    np.save(f'price/{date}.npy', price_tensor)
    np.save(f'text/{date}.npy', tweet_tensor)
    np.save(f'label/{date}.npy', label_tensor)

print("✅ Dataset prepared.")