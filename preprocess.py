import json
import re
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
tweets_raw = os.path.join(curr_dir, "raw-dataset", "tweets", "raw")
os.makedirs("preprocessed", exist_ok=True)
output_folder = os.path.join(curr_dir, "preprocessed")
output_folder_tweets = os.path.join(output_folder, "tweets")
os.makedirs(output_folder_tweets, exist_ok=True)

def preprocess_tweet(tweet_json):
    if "retweeted_status" in tweet_json:
        text = tweet_json["retweeted_status"].get("text", "")
    else:
        text = tweet_json.get("text", "")
    
    text = text.lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "AT_USER", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\$([a-zA-Z]+)", r"$ \1", text)
    text = re.sub(r"[^a-z0-9$><\- ]+", " ", text)
    tokens = text.split()

    return {
        "text": tokens,
        "created_at": tweet_json.get("created_at", ""),
        "user_id_str": tweet_json.get("user", {}).get("id_str", ""),
        "followers_count": tweet_json.get("user", {}).get("followers_count", 0)
    }

def preprocess_tweets_folder(input_path, output_folder):
    for stock in os.listdir(input_path):
        stock_inp_path = os.path.join(input_path, stock)
        stock_out_path = os.path.join(output_folder, stock)
        os.makedirs(stock_out_path, exist_ok=True)

        for filename in os.listdir(stock_inp_path):
            inp_file = os.path.join(stock_inp_path, filename)
            out_file = os.path.join(stock_out_path, filename)

            processed_tweets = []
            try:
                with open(inp_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            tweet = json.loads(line)
                            processed = preprocess_tweet(tweet)
                            processed_tweets.append(processed)
                        except json.JSONDecodeError as e:
                            print(f"Skipping line in {inp_file}: {e}")

                with open(out_file, "w") as f_out:
                    json.dump(processed_tweets, f_out, indent=2)
            except Exception as e:
                print(f"Error processing {inp_file}: {e}")

preprocess_tweets_folder(tweets_raw, output_folder_tweets)
