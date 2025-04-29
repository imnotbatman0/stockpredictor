# preprocess_prices.py

import os
import numpy as np
import pandas as pd
import pickle

PRICE_FEATURES = ['open', 'high', 'low', 'close']
MARGIN = 0.005  # 0.5%
WINDOW_SIZE = 5

def preprocess_prices(stock_folder):
    price_data = {}
    for fname in os.listdir(stock_folder):
        if fname.endswith('.txt'):
            ticker = fname.replace('.txt', '')
            df = pd.read_csv(os.path.join(stock_folder, fname), header=None)
            df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df = df[['date', 'open', 'high', 'low', 'close']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.reset_index(drop=True)

            # Normalize by previous close
            prev_close = df['close'].shift(1)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] / prev_close

            df = df.dropna().reset_index(drop=True)

            # Generate samples
            samples = []
            labels = []
            dates = []
            for i in range(WINDOW_SIZE, len(df)-1):
                window = df.loc[i-WINDOW_SIZE:i-1, ['open', 'high', 'low', 'close']].values
                today_close = df.loc[i-1, 'close']
                next_close = df.loc[i, 'close']
                pct_change = (next_close - today_close) / today_close
                if pct_change > MARGIN:
                    label = 1
                elif pct_change < -MARGIN:
                    label = 0
                else:
                    continue
                samples.append(window)
                labels.append(label)
                dates.append(df.loc[i, 'date'].strftime('%Y-%m-%d'))

            price_data[ticker] = {'samples': samples, 'labels': labels, 'dates': dates}

    return price_data
