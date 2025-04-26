import os
import pandas as pd
from datetime import datetime
import shutil

RAW_PRICES_DIR = 'raw-dataset/prices/raw'
RAW_TWEETS_DIR = 'preprocessed/tweets'
PROCESSED_DIR = 'preprocessed_data'

output_stock_dir = os.path.join(PROCESSED_DIR, "stocks")
output_tweet_dir = os.path.join(PROCESSED_DIR, "tweets")

START_DATE = pd.to_datetime("2014-01-01")
END_DATE = pd.to_datetime("2016-01-01")

def load_stock_data(stock_file):
    """Load stock data from a CSV file."""
    try:
        df = pd.read_csv(stock_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        return df
    except Exception as e:
        print(f"Error loading stock data for {stock_file}: {e}")
        return None

def get_valid_companies(stock_dir):
    """Identify companies with stock data starting before 2014-01-01."""
    valid_companies = []
    
    if not os.path.exists(stock_dir):
        print(f"Stock directory {stock_dir} does not exist.")
        return valid_companies
    
    for file in os.listdir(stock_dir):
        if file.endswith(".csv"):
            company = file.replace(".csv", "")
            stock_path = os.path.join(stock_dir, file)
            df = load_stock_data(stock_path)
            if df is None or df.empty:
                continue
            start_date = df['Date'].min()
            if start_date <= START_DATE:
                valid_companies.append(company)
    
    print(f"Found {len(valid_companies)} valid companies: {valid_companies}")
    return valid_companies

def parse_tweet_filename(filename):
    """Parse tweet filename to datetime."""
    try:
        return pd.to_datetime(filename, format="%Y-%m-%d")
    except ValueError as e:
        print(f"Failed to parse {filename}: {e}")
        return None

def save_stock_data(company, stock_df, output_stock_file):
    """Save stock data within the date range to the company's stock output file without header."""
    try:
        filtered_df = stock_df[stock_df['Date'].between(START_DATE, END_DATE)]
        if not filtered_df.empty:
            # Save without header and index
            filtered_df.to_csv(output_stock_file, index=False, header=False, sep=',')
        else:
            print(f"No stock data in range for {company}.")
    except Exception as e:
        print(f"Error saving stock data for {company}: {e}")

def copy_tweet_data(company, tweet_dir, output_tweet_dir, start_date, end_date):
    """Copy tweet data within the date range to the output directory."""
    tweet_subdir = os.path.join(tweet_dir, company)
    output_subdir = os.path.join(output_tweet_dir, company)
    
    if not os.path.exists(tweet_subdir):
        print(f"No tweet subdirectory for {company} at {tweet_subdir}.")
        return
    
    os.makedirs(output_subdir, exist_ok=True)
    
    for tweet_file in os.listdir(tweet_subdir):
        date = parse_tweet_filename(tweet_file)
        if date is not None and start_date <= date <= end_date:
            input_tweet_file = os.path.join(tweet_subdir, tweet_file)
            output_tweet_file = os.path.join(output_subdir, tweet_file)
            try:
                shutil.copy(input_tweet_file, output_tweet_file)
            except Exception as e:
                print(f"Error copying tweet file {tweet_file} for {company}: {e}")

def main():
    os.makedirs(output_stock_dir, exist_ok=True)
    os.makedirs(output_tweet_dir, exist_ok=True)
    
    valid_companies = get_valid_companies(RAW_PRICES_DIR)
    if not valid_companies:
        print("No valid companies found.")
        return
    
    for company in valid_companies:
        stock_path = os.path.join(RAW_PRICES_DIR, f"{company}.csv")
        stock_df = load_stock_data(stock_path)
        if stock_df is None:
            continue
        
        output_stock_file = os.path.join(output_stock_dir, f"{company}.txt")
        
        save_stock_data(company, stock_df, output_stock_file)
        
        copy_tweet_data(company, RAW_TWEETS_DIR, output_tweet_dir, START_DATE, END_DATE)
    
    print(f"Preprocessing complete. Data saved in {PROCESSED_DIR}")

if __name__ == "__main__":
    main()