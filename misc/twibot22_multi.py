import pandas as pd
import json
import glob
from datetime import datetime
from multiprocessing import Pool, current_process
import logging
 
# Set up logging
logging.basicConfig(filename='twibot22_preprocessing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
 
def process_tweet_file(file, user_ids, user_indices):
    process_id = current_process().pid
    print(f"Processing {file} with PID {process_id}\n")
    logging.info(f"Processing {file} with PID {process_id}\n")
    # Load tweet file
    with open(file, 'r') as f:
        tweets_ = json.load(f)
    # Normalize tweet columns
    tweets = pd.json_normalize(tweets_, sep='_')
 
    # Flatten 'entities_user_mentions'
    if 'entities_user_mentions' in tweets.columns:
        user_mentions = tweets.explode('entities_user_mentions')
        user_mentions = pd.json_normalize(user_mentions['entities_user_mentions']).add_prefix('user_mentions_')
        tweets = tweets.drop(columns=['entities_user_mentions']).join(user_mentions)
 
    # Flatten 'entities_media'
    if 'entities_media' in tweets.columns:
        media = tweets.explode('entities_media')
        media = pd.json_normalize(media['entities_media']).add_prefix('media_')
        tweets = tweets.drop(columns=['entities_media']).join(media)
 
    results = []
    # Process each tweet
    for index, tweet in tweets.iterrows():
        try:
            author_id = str(tweet['author_id'])
            if author_id in user_ids:
                author_index = user_indices[author_id]
                if tweet['referenced_tweets']:
                    #print(f'tweet {index} from author {user_id} has referenced tweets')
                    for ref_tweet in tweet['referenced_tweets']:
                        ref_type = ref_tweet['type']
                        ref_user_id = str(ref_tweet['id'])
                        if ref_type == 'retweeted':
                            results.append((author_index, 'type2', 1))
                        elif ref_type == 'replied_to' or ref_type == 'quoted':
                            results.append((author_index, 'type3', 1))
                        if ref_user_id in user_ids:
                            ref_user_index = user_indices[ref_user_id]
                            results.append((ref_user_index, 'inf', 1))
                else:
                    #print(f'tweet {index} has no referenced tweets')
                    results.append((author_index, 'type1', 1))
                    results.append((author_index, 'inf', tweet['public_metrics_retweet_count'] + tweet['public_metrics_like_count']))
 
                # Add influence score for mentions
                mentioned_users = tweet.get('user_mentions_id', [])
                if not pd.isna(mentioned_users):
                    #print(f'tweet {index} has mentioned users')
                    if not isinstance(mentioned_users, list):
                        mentioned_users = [mentioned_users]
                    for mentioned_user_id in mentioned_users:
                        mentioned_user_id = str(mentioned_user_id)
                        if mentioned_user_id in user_ids:
                            mentioned_user_index = user_indices[mentioned_user_id]
                            results.append((mentioned_user_index, 'inf', 1))
                else:
                    print(f'tweet {index} has no mentioned users')
 
        except KeyError as e:
            print(f"KeyError: {e} in file {file} at index {index}")
            print(tweet)
 
    print(f"Finished processing {file} with PID {process_id}\n")
    logging.info(f"Finished processing {file} with PID {process_id}\n")
    return results
 
def preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path):
 
    # Load users data
    users = pd.read_json(user_data_path)
    print("Loaded user data\n")
 
    # Load labels
    labels = pd.read_csv(label_data_path)
    print("Loaded labels\n")
 
    # Create DataFrame
    preprocessed_data = pd.DataFrame()
 
    # Create ID Column
    preprocessed_data['userid'] = users['id']
    print("Created ID Column\n")
 
    # Merge with labels to create label column
    preprocessed_data = preprocessed_data.merge(labels[['id', 'label']], left_on='userid', right_on='id', how='left').drop(columns=['id'])
    print("Created label column\n")
 
    # Clean userid column
    preprocessed_data['userid'] = preprocessed_data['userid'].str.replace('u', '')
 
    # Create ff Column
    preprocessed_data['ff'] = users['public_metrics'].apply(lambda x: x['followers_count'] / (x['following_count'] + 1))
    print("Created ff column\n")
 
    # Initialize tweet type counts and influence scores
    preprocessed_data['type1'] = 0  # Original tweets
    preprocessed_data['type2'] = 0  # Retweets
    preprocessed_data['type3'] = 0  # Comments
    preprocessed_data['inf'] = 0
 
    # Create a dictionary of user ids and their indices for quick lookup
    user_ids = set(preprocessed_data['userid'])
    user_indices = {user_id: idx for idx, user_id in enumerate(preprocessed_data['userid'])}
 
    # Load and process tweet data in parallel
    tweet_files = glob.glob(tweet_data_paths)
    with Pool() as pool:
        results = pool.starmap(process_tweet_file, [(file, user_ids, user_indices) for file in tweet_files])
 
    # Apply results to preprocessed_data
    for result in results:
        for author_index, col, value in result:
            preprocessed_data.at[author_index, col] += value
 
    print("Finished tweet processing\n")
 
    # Normalize tweet type counts
    type_sum = preprocessed_data[['type1', 'type2', 'type3']].sum(axis=1)
    preprocessed_data['type1'] = preprocessed_data['type1'] / type_sum
    preprocessed_data['type2'] = preprocessed_data['type2'] / type_sum
    preprocessed_data['type3'] = preprocessed_data['type3'] / type_sum
    print("Normalized tweet data\n")
 
    # Normalize influence scores
    max_inf = preprocessed_data['inf'].max()
    if max_inf > 0:
        preprocessed_data['inf'] = preprocessed_data['inf'] / max_inf
    print("Normalized influence scores\n")
 
    # Split the dataset into human and bot subsets
    human_subset = preprocessed_data[preprocessed_data['label'] == 'human']
    bot_subset = preprocessed_data[preprocessed_data['label'] == 'bot']
    print("Split the dataset\n")
 
    # Save preprocessed user data to CSV
    preprocessed_data.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    human_subset.to_csv('preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    bot_subset.to_csv('preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
 
if __name__ == "__main__":
    user_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/user.json'  # Replace with the actual path
    label_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/label.csv'
    tweet_data_paths = '/cs/student/projects1/sec/2023/jmoussa/twibot22/tweet/tweet_*.json'
    output_path = 'preprocessed_twibot22_users.csv'  # Replace with the desired output path
    start_time = datetime.now().strftime("%D %H:%M:%S")
    print(f'Start time is {start_time} \n')
    preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path)
    print(f'Start time was {start_time} \n')
    print(f'End time is {datetime.now().strftime("%D %H:%M:%S")}\n')