# prepare_dataset.py (revised for batching by date)

import os
import numpy as np
from preprocess_prices import preprocess_prices
from preprocess_tweets import preprocess_tweets

price_folder = 'data/stocks/'
tweet_folder = 'data/tweets/'

os.makedirs('price', exist_ok=True)
os.makedirs('text', exist_ok=True)
os.makedirs('label', exist_ok=True)

print("Processing prices...")
price_data = preprocess_prices(price_folder)  # {ticker: {'samples', 'labels', 'dates'}}
print("Processing tweets...")
tweet_data = preprocess_tweets(tweet_folder, {t: d['dates'] for t, d in price_data.items()})

# Build master list of aligned dates
all_dates = sorted(list(set.intersection(*[set(price_data[t]['dates']) for t in price_data])))

for date_idx, date in enumerate(all_dates):
    price_tensor = []
    tweet_tensor = []
    label_tensor = []
    for ticker in price_data:
        idx = price_data[ticker]['dates'].index(date)
        price_tensor.append(price_data[ticker]['samples'][idx])
        tweet_tensor.append(tweet_data[ticker].get(date, np.zeros((5, 10, 64))))
        label_tensor.append(price_data[ticker]['labels'][idx])

    price_tensor = np.stack(price_tensor)  # [num_stocks, 5, 4]
    tweet_tensor = np.stack(tweet_tensor)  # [num_stocks, 5, 10, 64]
    label_tensor = np.array(label_tensor)  # [num_stocks]

    np.save(f'price/{date}.npy', price_tensor)
    np.save(f'text/{date}.npy', tweet_tensor)
    np.save(f'label/{date}.npy', label_tensor)

print("Done. Total days:", len(all_dates))

