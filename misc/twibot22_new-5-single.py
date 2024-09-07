import pandas as pd
import json
import glob
from datetime import datetime

def process_tweet_file(file):
    try:
        print(f"Processing file: {file}")
        with open(file, 'r') as f:
            tweets_ = json.load(f)
        tweets = pd.json_normalize(tweets_, sep='_')
        
        tweets['author_id'] = tweets['author_id'].astype(str)
        
        print(f"Sample of 'referenced_tweets' before processing: {tweets['referenced_tweets'].head()}")
        if 'referenced_tweets' in tweets.columns:
            referenced_tweets = tweets['referenced_tweets'].apply(lambda x: any(d['type'] == 'retweeted' for d in x) if isinstance(x, list) else False)
            retweets = tweets[referenced_tweets]
            referenced_tweets = tweets['referenced_tweets'].apply(lambda x: any(d['type'] == 'replied_to' or d['type'] == 'quoted' for d in x) if isinstance(x, list) else False)
            replies = tweets[referenced_tweets]
            originals = tweets[tweets['referenced_tweets'].isna()]
        else:
            originals = tweets
            retweets = pd.DataFrame()
            replies = pd.DataFrame()
        
        originals = originals.copy()
        retweets = retweets.copy()
        replies = replies.copy()
        
        # Sum the influence scores for original tweets
        originals['inf'] = originals['public_metrics_retweet_count'].fillna(0) + \
                           originals['public_metrics_like_count'].fillna(0) + \
                           originals['public_metrics_reply_count'].fillna(0)
        
        # Debugging output to check influence scores calculation
        print(f"Originals influence scores:\n{originals[['author_id', 'inf']].head()}")

        originals['type1'] = 1
        retweets['type2'] = 1
        replies['type3'] = 1
        
        all_tweets = pd.concat([originals, retweets, replies])
        all_tweets = all_tweets[['author_id', 'type1', 'type2', 'type3', 'inf']]
        all_tweets = all_tweets.groupby('author_id').sum().reset_index()
        
        # Debugging output to check combined data
        print(f"Combined data before merging original tweet count:\n{all_tweets.head()}")

        # Add count of original tweets
        originals_count = originals.groupby('author_id').size().reset_index(name='original_tweet_count')
        result = pd.merge(all_tweets, originals_count, on='author_id', how='left').fillna(0)
        result['author_id'] = result['author_id'].astype(str)
        
        # Debugging output to check final result
        print(f"Final result:\n{result.head()}")

        print(f"Processed {file} successfully with {len(result)} records.")
        return result
    
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return pd.DataFrame()

def preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path):
    users = pd.read_json(user_data_path)
    print("Loaded user data\n")
    labels = pd.read_csv(label_data_path)
    print("Loaded labels\n")
    
    preprocessed_data = pd.DataFrame()
    preprocessed_data['userid'] = users['id'].astype(str)
    print("Created ID Column\n")
    preprocessed_data = preprocessed_data.merge(labels[['id', 'label']], left_on='userid', right_on='id', how='left').drop(columns=['id'])
    print("Created label column\n")
    preprocessed_data['userid'] = preprocessed_data['userid'].str.replace('u', '').astype(str)
    preprocessed_data['ff'] = users['public_metrics'].apply(lambda x: x['followers_count'] / (x['following_count'] + 1))
    print("Created ff column\n")

    tweet_files = glob.glob(tweet_data_paths)
    print(f"Found {len(tweet_files)} tweet files\n")
    
    results = []
    for file in tweet_files:
        result = process_tweet_file(file)
        results.append(result)
    
    final_results = pd.concat(results).groupby('author_id').sum().reset_index()
    final_results = final_results.rename(columns={'author_id': 'userid'})
    
    # Debugging output to check final combined results
    print(f"Final combined results:\n{final_results.head()}")

    preprocessed_data = pd.merge(preprocessed_data, final_results, on='userid', how='left').fillna(0)
    preprocessed_data.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3', 'original_tweet_count'], index=False)

if __name__ == "__main__":
    user_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/user.json'  # Replace with the actual path
    label_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/label.csv'
    tweet_data_paths = '/cs/student/projects1/sec/2023/jmoussa/twibot22/tweet/tweet_8.json'
    output_path = 'output2/preprocessed_twibot22_8.csv'  # Replace with the desired output path
    start_time = datetime.now().strftime("%D %H:%M:%S")
    print(f'Start time is {start_time} \n')
    preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path)
    print(f'Start time was {start_time} \n')
    print(f'End time is {datetime.now().strftime("%D %H:%M:%S")} \n')
