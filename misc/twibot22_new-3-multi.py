import pandas as pd
import json
import glob
from datetime import datetime
from multiprocessing import Pool

def process_tweet_file(file):
    try:
        print(f"Processing file: {file}")
        with open(file, 'r') as f:
            tweets_ = json.load(f)
        tweets = pd.json_normalize(tweets_, sep='_')
        
        #print(f"Initial columns in tweets: {tweets.columns}")
        
        if 'entities_user_mentions' in tweets.columns:
            user_mentions = tweets.explode('entities_user_mentions')
            user_mentions = pd.json_normalize(user_mentions['entities_user_mentions']).add_prefix('user_mentions_')
            tweets = tweets.drop(columns=['entities_user_mentions']).join(user_mentions)
        
        #print(f"Columns after processing user mentions: {tweets.columns}")
        
        tweets['author_id'] = tweets['author_id'].astype(str)

        # Debugging statement to understand the structure of 'referenced_tweets'
        print(f"Sample of 'referenced_tweets' before processing: {tweets['referenced_tweets'].head()}")

        # Check for 'referenced_tweets' and process
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

        # Ensure we have a copy to avoid SettingWithCopyWarning
        originals = originals.copy()
        retweets = retweets.copy()
        replies = replies.copy()

        originals['type1'] = 1
        originals['inf'] = originals['public_metrics_retweet_count'] + originals['public_metrics_like_count']
        
        retweets['type2'] = 1
        retweets['inf'] = retweets['public_metrics_retweet_count'] + retweets['public_metrics_like_count']

        replies['type3'] = 1
        replies['inf'] = replies['public_metrics_retweet_count'] + replies['public_metrics_like_count']
        
        all_tweets = pd.concat([originals, retweets, replies])
        all_tweets = all_tweets[['author_id', 'type1', 'type2', 'type3', 'inf']]
        all_tweets = all_tweets.groupby('author_id').sum().reset_index()
        
        if 'user_mentions_id' in tweets.columns:
            mentions = tweets[['user_mentions_id']].dropna(subset=['user_mentions_id'])
            mentions = mentions.explode('user_mentions_id')
            mentions['user_mentions_id'] = mentions['user_mentions_id'].astype(str)
            mentions['inf'] = 1  # Increment influence score by 1 for each mention
            mentions = mentions[['user_mentions_id', 'inf']].groupby('user_mentions_id').sum().reset_index()
            mentions.columns = ['author_id', 'inf_mentions']
        
            result = pd.merge(all_tweets, mentions, on='author_id', how='left')
            result['inf'] += result['inf_mentions'].fillna(0)
            result = result.drop(columns=['inf_mentions'])
        else:
            result = all_tweets
        
        if 'referenced_tweets' in tweets.columns:
            references = tweets[tweets['referenced_tweets'].notna()].explode('referenced_tweets')
            if not references.empty:
                references = pd.json_normalize(references['referenced_tweets']).add_prefix('referenced_')
                references = references.dropna(subset=['referenced_id'])
                references['referenced_id'] = references['referenced_id'].astype(str)
                references['inf'] = 1  # Increment influence score by 1 for each reference
                references = references[['referenced_id', 'inf']].groupby('referenced_id').sum().reset_index()
                references.columns = ['author_id', 'inf_references']
                result = pd.merge(result, references, on='author_id', how='left')
                result['inf'] += result['inf_references'].fillna(0)
                result = result.drop(columns=['inf_references'])
        
        result['author_id'] = result['author_id'].astype(str)
        
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
    
    with Pool() as pool:
        results = pool.map(process_tweet_file, tweet_files)
    
    final_results = pd.concat(results).groupby('author_id').sum().reset_index()
    final_results = final_results.rename(columns={'author_id': 'userid'})
    
    preprocessed_data = pd.merge(preprocessed_data, final_results, on='userid', how='left').fillna(0)
    
    #type_sum = preprocessed_data[['type1', 'type2', 'type3']].sum(axis=1)
    #preprocessed_data[['type1', 'type2', 'type3']] = preprocessed_data[['type1', 'type2', 'type3']].div(type_sum, axis=0)
    #print("Normalized tweet data\n")
    
    #max_inf = preprocessed_data['inf'].max()
    #if max_inf > 0:
    #    preprocessed_data['inf'] = preprocessed_data['inf'] / max_inf
    #print("Normalized influence scores\n")
    
    #human_subset = preprocessed_data[preprocessed_data['label'] == 'human']
    #bot_subset = preprocessed_data[preprocessed_data['label'] == 'bot']
    #print("Split the dataset\n")
    
    preprocessed_data.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    #human_subset.to_csv('preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    #bot_subset.to_csv('preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)

if __name__ == "__main__":
    user_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/user.json'  # Replace with the actual path
    label_data_path = '/cs/student/projects1/sec/2023/jmoussa/twibot22/label.csv'
    tweet_data_paths = '/cs/student/projects1/sec/2023/jmoussa/twibot22/tweet/tweet_8.json'
    output_path = 'output/preprocessed_twibot22_8.csv'  # Replace with the desired output path
    start_time = datetime.now().strftime("%D %H:%M:%S")
    print(f'Start time is {start_time} \n')
    preprocess_twibot22(user_data_path, label_data_path, tweet_data_paths, output_path)
    print(f'Start time was {start_time} \n')
    print(f'End time is {datetime.now().strftime("%D %H:%M:%S")} \n')
