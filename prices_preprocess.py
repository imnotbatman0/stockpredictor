import os
import pandas as pd
from datetime import datetime
import shutil

# Relative directory paths
RAW_PRICES_DIR = 'raw-dataset/prices/raw'
RAW_TWEETS_DIR = 'preprocessed/tweets'
PROCESSED_DIR = 'preprocessed_data'

# Output directories
output_stock_dir = os.path.join(PROCESSED_DIR, "stocks")
output_tweet_dir = os.path.join(PROCESSED_DIR, "tweets")

# Thresholds for labeling
POSITIVE_THRESHOLD = 0.55  # Percentage change >= 0.55% for +1
NEGATIVE_THRESHOLD = -0.5  # Percentage change <= -0.5% for -1

def load_stock_data(stock_file):
    """Load stock data from a CSV file."""
    try:
        df = pd.read_csv(stock_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except Exception:
        return None

def get_valid_companies(stock_dir):
    """Identify companies with stock data starting before 2014-01-01."""
    valid_companies = []
    company_start_dates = {}
    
    if not os.path.exists(stock_dir):
        return valid_companies, company_start_dates
    
    for file in os.listdir(stock_dir):
        if file.endswith(".csv"):
            company = file.replace(".csv", "")
            stock_path = os.path.join(stock_dir, file)
            df = load_stock_data(stock_path)
            if df is None or df.empty:
                continue
            start_date = df['Date'].min()
            if start_date <= pd.to_datetime("2014-01-01"):
                valid_companies.append(company)
                company_start_dates[company] = start_date
    
    return valid_companies, company_start_dates

def parse_tweet_filename(filename):
    """Parse tweet filename to datetime."""
    try:
        return pd.to_datetime(filename, format="%Y-%m-%d")
    except ValueError:
        return None

def get_company_dates(company, stock_dir, tweet_dir):
    """Get dates with both stock and tweet data for a single company."""
    # Load stock data
    stock_path = os.path.join(stock_dir, f"{company}.csv")
    stock_df = load_stock_data(stock_path)
    if stock_df is None or stock_df.empty:
        return set()
    
    stock_dates = set(stock_df['Date'])
    
    # Get tweet dates
    tweet_subdir = os.path.join(tweet_dir, company)
    if not os.path.exists(tweet_subdir):
        return set()
    
    tweet_files = [f for f in os.listdir(tweet_subdir)]
    tweet_dates = set()
    for tweet_file in tweet_files:
        date = parse_tweet_filename(tweet_file)
        if date is not None:
            file_path = os.path.join(tweet_subdir, tweet_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1)
                tweet_dates.add(date)
            except Exception:
                continue
    
    # Intersection of stock and tweet dates
    common_dates = stock_dates.intersection(tweet_dates)
    return common_dates

def calculate_price_change(stock_df, date, prev_date):
    """Calculate percentage change in closing price and assign label."""
    if prev_date not in stock_df['Date'].values:
        return None
    
    current_close = stock_df[stock_df['Date'] == date]['Close'].iloc[0]
    prev_close = stock_df[stock_df['Date'] == prev_date]['Close'].iloc[0]
    
    if prev_close == 0:
        return None
    
    pct_change = ((current_close - prev_close) / prev_close) * 100
    
    if pct_change >= POSITIVE_THRESHOLD:
        return 1
    elif pct_change <= NEGATIVE_THRESHOLD:
        return -1
    else:
        return 0

def save_stock_data(company, date, stock_data, label, output_stock_file):
    """Append stock data without label to the company's stock output file."""
    try:
        with open(output_stock_file, 'a') as f:
            f.write(f"{date.date()},{stock_data['Open']},{stock_data['High']},"
                    f"{stock_data['Low']},{stock_data['Close']},"
                    f"{stock_data['Adj Close']},{stock_data['Volume']}\n")
    except Exception:
        pass

def copy_tweet_data(company, date, input_tweet_file, output_tweet_file):
    """Copy tweet data to the output directory."""
    try:
        os.makedirs(os.path.dirname(output_tweet_file), exist_ok=True)
        shutil.copy(input_tweet_file, output_tweet_file)
    except Exception:
        pass

def main():
    # Create output directories
    os.makedirs(output_stock_dir, exist_ok=True)
    os.makedirs(output_tweet_dir, exist_ok=True)
    
    # Step 1: Identify valid companies
    valid_companies, company_start_dates = get_valid_companies(RAW_PRICES_DIR)
    if not valid_companies:
        return
    
    # Step 2: Process each company independently
    for company in valid_companies:
        stock_path = os.path.join(RAW_PRICES_DIR, f"{company}.csv")
        stock_df = load_stock_data(stock_path)
        if stock_df is None or stock_df.empty:
            continue
        
        # Get dates with both stock and tweet data for this company
        common_dates = get_company_dates(company, RAW_PRICES_DIR, RAW_TWEETS_DIR)
        if not common_dates:
            continue
        
        # Initialize stock output file without header
        output_stock_file = os.path.join(output_stock_dir, f"{company}.txt")
        
        # Step 3: Process each date
        common_dates = sorted(common_dates)
        for i, date in enumerate(common_dates):
            if i == 0:
                continue  # Skip first date (no previous close)
            
            prev_date = common_dates[i-1]
            
            # Calculate price change and label
            label = calculate_price_change(stock_df, date, prev_date)
            
            # Skip if label is 0 (neutral) or None
            if label == 0 or label is None:
                continue
            
            # Get stock data for the date
            stock_data = stock_df[stock_df['Date'] == date]
            if stock_data.empty:
                continue
            
            # Save stock data
            save_stock_data(company, date, stock_data.iloc[0], label, output_stock_file)
            
            # Copy tweet data
            tweet_filename = f"{date.date()}"
            input_tweet_file = os.path.join(RAW_TWEETS_DIR, company, tweet_filename)
            output_tweet_file = os.path.join(output_tweet_dir, company, tweet_filename)
            
            if os.path.exists(input_tweet_file):
                copy_tweet_data(company, date, input_tweet_file, output_tweet_file)
    
    print(f"Preprocessing complete. Data saved in {PROCESSED_DIR}")

if __name__ == "__main__":
    main()