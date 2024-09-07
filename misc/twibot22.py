import pandas as pd
import json
import glob
import ijson
from datetime import datetime

def preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path):

    # Load users data
    users = pd.read_json(user_data_path)
    print("Loaded user data\n")
    #print(users.head())  # Print the first few rows to check if it loads correctly

    # Load labels
    labels = pd.read_csv(label_data_path)
    print("Loaded labels\n")

    # Create Dataframe
    preprocessed_data = pd.DataFrame()

    # Create ID Column
    preprocessed_data['userid'] = users['id']
    print("Created ID Column\n")

    # Verify columns before processing tweets
    #print("Columns in preprocessed_data before merge:")
    #print(preprocessed_data.columns)

    # Merge with labels to create label column
    preprocessed_data = preprocessed_data.merge(labels[['id', 'label']], left_on='userid', right_on='id', how='left').drop(columns=['id'])
    print("Created label column\n")
    #print(preprocessed_data.head())  # Print the first few rows to check if it loads correctly

    preprocessed_data['userid'] = users['id'].str.replace('u', '')

    # Create ff Column
    preprocessed_data['ff'] = users['public_metrics'].apply(lambda x: x['followers_count'] / (x['following_count'] + 1))
    print("Created ff column\n")

    # Initialize tweet type counts and influence scores
    preprocessed_data['type1'] = 0  # Original tweets
    preprocessed_data['type2'] = 0  # Retweets
    preprocessed_data['type3'] = 0  # Comments
    preprocessed_data['inf'] = 0

    #print(preprocessed_data)

    # Load and combine tweet data without chunks
    #tweet_files = glob.glob(tweet_data_paths)
    #tweets = pd.concat([pd.read_json(file, lines=True) for file in tweet_files])
    #print(tweets.head())  # Print the first few rows to check if it loads correctly 

    #tweets = pd.read_json(tweet_data_paths, lines=True, nrows=10)
    tweet_files = glob.glob(tweet_data_paths)
    for file in tweet_files:
        with open(file, 'r') as f:
            tweets_ = json.load(f)
        tweets = pd.json_normalize(tweets_, sep='_')
        #print(f"Columns in tweet data: {tweets.columns}\n")  # Print columns to verify structure
        #print(tweets.head())  # Print the first few rows to check the structure

        # Inspect nested columns to ensure they are fully flattened
        #if 'entities_user_mentions' in tweets.columns:
            #print("\nEntities User Mentions:")
            #print(tweets['entities_user_mentions'].head())

        #if 'entities_media' in tweets.columns:
            #print("\nEntities Media:")
            #print(tweets['entities_media'].head())

        # Flatten 'entities_user_mentions'
        if 'entities_user_mentions' in tweets.columns:
            user_mentions = tweets.explode('entities_user_mentions')
            user_mentions = pd.json_normalize(user_mentions['entities_user_mentions']).add_prefix('user_mentions_')
            tweets = tweets.drop(columns=['entities_user_mentions']).join(user_mentions)
            #print("\nFlattened User Mentions Columns:")
            #print(tweets.columns)

        # Flatten 'entities_media'
        if 'entities_media' in tweets.columns:
            media = tweets.explode('entities_media')
            media = pd.json_normalize(media['entities_media']).add_prefix('media_')
            tweets = tweets.drop(columns=['entities_media']).join(media)
            #print("\nFlattened Media Columns:")
            #print(tweets.columns)

        #print("\nFinal tweet data columns:")
        #print(tweets.columns)
        #print(tweets.head())

    # Process each tweet
        for index, tweet in tweets.iterrows():
            try:
                user_id = str(tweet['author_id'])
                #print(f"inside try for tweet {index} with id {user_id}")
                if user_id in preprocessed_data['userid'].values:
                    user_index = preprocessed_data.index[preprocessed_data['userid'] == user_id][0]
                    if tweet['referenced_tweets']:
                        print(f'inside if statement of referenced tweets for tweet {index}')
                        if tweet['referenced_tweets'][0]['type'] == 'retweeted':
                            preprocessed_data.at[user_index, 'type2'] += 1
                        elif tweet['referenced_tweets'][0]['type'] == 'replied_to':
                            preprocessed_data.at[user_index, 'type3'] += 1
                    else:
                        print(f'inside else statement for tweet {index}')
                        preprocessed_data.at[user_index, 'type1'] += 1
                        preprocessed_data.at[user_index, 'inf'] += tweet['public_metrics_retweet_count'] + tweet['public_metrics_like_count']
                else:
                    print(f'author_id {user_id} of tweet {index} not found in users list')
            except KeyError as e:
                print(f"KeyError: {e} in file {tweet_data_paths} at index {index}")
                print(tweet)
        print("Finished tweet processing\n")
        #print(preprocessed_data.head())

    # Normalize tweet type counts
    type_sum = preprocessed_data[['type1', 'type2', 'type3']].sum(axis=1)
    preprocessed_data['type1'] = preprocessed_data['type1'] / type_sum
    preprocessed_data['type2'] = preprocessed_data['type2'] / type_sum
    preprocessed_data['type3'] = preprocessed_data['type3'] / type_sum
    print("Normalized tweet data\n")

    # Split the dataset into human and bot subsets
    human_subset = preprocessed_data[preprocessed_data['label'] == 'human']
    bot_subset = preprocessed_data[preprocessed_data['label'] == 'bot']

    # Verify columns before saving
    #print("Columns in preprocessed_data before saving:")
    #print(preprocessed_data.columns)
    
    # Save full preprocessed data to CSV
    preprocessed_data.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    
    human_subset.to_csv('preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    bot_subset.to_csv('preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    
if __name__ == "__main__":
    user_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/user.json'  # Replace with the actual path
    label_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/label.csv'
    tweet_data_paths = '/cs/student/projects1/sec/2023/jmoussa/twibot22/tweet/tweet_*.json'
    output_path = 'preprocessed_twibot22_combined.csv'  # Replace with the desired output path
    print(f'Start time is {datetime.now().strftime("%D %H:%M:%S")}')
    preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path)
    print(f'End time is {datetime.now().strftime("%D %H:%M:%S")}')
