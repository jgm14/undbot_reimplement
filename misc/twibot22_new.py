import pandas as pd
import json
import glob
from datetime import datetime
 
def preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path):
 
    # Load users data
    users = pd.read_json(user_data_path)
    print("Loaded user data\n")
 
    # Load labels
    labels = pd.read_csv(label_data_path)
    print("Loaded labels\n")
 
    # Create Dataframe
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

    referenced_tweets_counter = 0
 
    # Load and process tweet data
    tweet_files = glob.glob(tweet_data_paths)
    for file in tweet_files:
        print(f"Processing {file}\n")
        with open(file, 'r') as f:
            tweets_ = json.load(f)
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
 
        # Process each tweet
        for index, tweet in tweets.iterrows():
            try:
                user_id = str(tweet['author_id'])
                if user_id in preprocessed_data['userid'].values:
                    user_index = preprocessed_data.index[preprocessed_data['userid'] == user_id][0]
 
                    # Add to type counts and influence scores
                    if tweet['referenced_tweets']:
                        print(f'tweet {index} from author {user_id} has referenced tweets')
                        referenced_tweets_counter+=1
                        if tweet['referenced_tweets'][0]['type'] == 'retweeted':
                            preprocessed_data.at[user_index, 'type2'] += 1
                        elif tweet['referenced_tweets'][0]['type'] == 'replied_to':
                            preprocessed_data.at[user_index, 'type3'] += 1
                            reply_to_user_id = str(tweet['in_reply_to_user_id'])
                            if reply_to_user_id in preprocessed_data['userid'].values:
                                reply_to_user_index = preprocessed_data.index[preprocessed_data['userid'] == reply_to_user_id][0]
                                preprocessed_data.at[reply_to_user_index, 'inf'] += 1
                    else:
                        print(f'tweet {index} has no referenced tweets')
                        preprocessed_data.at[user_index, 'type1'] += 1
                        preprocessed_data.at[user_index, 'inf'] += tweet['public_metrics_retweet_count'] + tweet['public_metrics_like_count']
 
                    # Add influence score for mentions
                    mentioned_users = tweet.get('user_mentions_id', [])
                    if not pd.isna(mentioned_users):
                        if not isinstance(mentioned_users, list):
                            mentioned_users = [mentioned_users]
                        for mentioned_user_id in mentioned_users:
                            mentioned_user_id = str(mentioned_user_id)
                            if mentioned_user_id in preprocessed_data['userid'].values:
                                mentioned_user_index = preprocessed_data.index[preprocessed_data['userid'] == mentioned_user_id][0]
                                preprocessed_data.at[mentioned_user_index, 'inf'] += 1
 
            except KeyError as e:
                print(f"KeyError: {e} in file {file} at index {index}")
                print(tweet)
 
        print(f'Finished processing file {file}')
 
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
    output_path = 'preprocessed_twibot22_combined.csv'  # Replace with the desired output path
    print(f'Start time is {datetime.now().strftime("%D %H:%M:%S")} \n')
    preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path)
    print(f'Number of referenced tweets is {referenced_tweets_counter} \n')
    print(f'End time is {datetime.now().strftime("%D %H:%M:%S")} \n')
