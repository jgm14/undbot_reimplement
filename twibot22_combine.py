import pandas as pd
import glob
import json

def combine_preprocessed_files(file_paths, output_path):
    combined_df = None
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        
        if combined_df is None:
            # Initialize combined_df with the first file, except for the columns to be summed
            combined_df = df.copy()
            combined_df[['retweet_count', 'like_count', 'reply_count', 'type1', 'type2', 'type3', 'original_tweet_count']] = 0  # Resetting the columns to be summed
        else:
            # Verify the structure
            if not all(combined_df.columns == df.columns):
                raise ValueError(f"Column mismatch found in file {file_path}")
        
        # Sum the columns that need to be aggregated across files
        combined_df['retweet_count'] += df['retweet_count'].fillna(0)
        combined_df['like_count'] += df['like_count'].fillna(0)
        combined_df['reply_count'] += df['reply_count'].fillna(0)
        combined_df['type1'] += df['type1']
        combined_df['type2'] += df['type2']
        combined_df['type3'] += df['type3']
        combined_df['original_tweet_count'] += df['original_tweet_count']
    
    print(f'Sample of raw combined dataframe: {combined_df.head()}')

    # Convert label column to numerical
    combined_df['label'] = combined_df['label'].apply(lambda x: 1 if x == 'bot' else 0)
    print("Converted labels to numerical values\n")
    
    # Derive the inf column according to Approach 1 (average retweets, likes, replies)
    combined_df['avg_retweets'] = combined_df.apply(lambda row: row['retweet_count'] / row['original_tweet_count'] if row['original_tweet_count'] > 0 else 0, axis=1)
    combined_df['avg_likes'] = combined_df.apply(lambda row: row['like_count'] / row['original_tweet_count'] if row['original_tweet_count'] > 0 else 0, axis=1)
    combined_df['avg_replies'] = combined_df.apply(lambda row: row['reply_count'] / row['original_tweet_count'] if row['original_tweet_count'] > 0 else 0, axis=1)
    
    # Calculate the total influence score (inf) as the sum of the average retweets, likes, and replies
    combined_df['inf'] = combined_df['avg_retweets'] + combined_df['avg_likes'] + combined_df['avg_replies']
    
    # Debugging output to check derived influence scores
    print(f"Derived influence scores (inf):\n{combined_df[['userid', 'inf']].head()}")

    # Normalize influence scores
    #max_inf = combined_df['inf'].max()
    #if max_inf > 0:
    #    combined_df['inf'] = combined_df['inf'] / max_inf
    #print("Normalized influence scores\n")
    
    # Normalize tweet type counts
    type_sum = combined_df[['type1', 'type2', 'type3']].sum(axis=1)
    combined_df[['type1', 'type2', 'type3']] = combined_df[['type1', 'type2', 'type3']].div(type_sum, axis=0).fillna(0)
    print("Normalized tweet type counts\n")

    #Create human and bot subsets
    human_subset = combined_df[combined_df['label'] == 0]
    bot_subset = combined_df[combined_df['label'] == 1]
    print("Split the dataset into human and bot subsets\n")
    
    # Save the combined DataFrame to CSV
    combined_df.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved full combined data to {output_path} with {len(combined_df)} records'\n")

    # Save the human and bot subsets to CSV
    #human_subset.to_csv('twibot-22/combined_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    #print(f"Saved full human subset \n")
    #bot_subset.to_csv('twibot-22/combined_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    #print(f"Saved full bot subset \n")

    return combined_df, human_subset, bot_subset

def sample_preprocessed_data(human_subset, bot_subset, human_sample_size, bot_sample_size):
    # Perform sampling to get the desired number of humans and bots
    sampled_humans = human_subset.sample(n=human_sample_size, random_state=1)
    sampled_bots = bot_subset.sample(n=bot_sample_size, random_state=1)
    print('Sampling the human and bot subsets \n')

    # Combine the sampled data
    sampled_data = pd.concat([sampled_humans, sampled_bots])

    # Save the combined sampled data to a new file
    sampled_data.to_csv('twibot-22/sample2/sampled_combined_preprocessed_twibot22.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved sampled combined data \n")

    # Save the individual sampled human and bot data
    sampled_humans.to_csv('twibot-22/sample2/sampled_combined_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved sampled human subset \n")
    sampled_bots.to_csv('twibot-22/sample2/sampled_combined_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved sampled bot subset \n")

def sample_user_community(combined_df, community_file):

   # Load the user1.json file that contains the userIDs for the specific subcommunity
    with open(community_file, 'r') as f:
        user_ids = json.load(f)

    # Remove the 'u' prefix from the user IDs
    user_ids = [int(uid.replace('u', '')) for uid in user_ids]
    print(user_ids[2:10])

    # Filter the combined dataframe to only include users in the subcommunity
    community_sample = combined_df[combined_df['userid'].isin(user_ids)]
    print(len(combined_df['userid']))

    # Save the subcommunity sample to a new file
    community_sample.to_csv('twibot-22/sample4/subcommunity_sampled_preprocessed_twibot22.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved subcommunity sampled data \n")

    #Create human and bot subsets
    community_sample_humans = community_sample[community_sample['label'] == 0]
    community_sample_bots = community_sample[community_sample['label'] == 1]

    # Save the individual sampled human and bot data
    community_sample_humans.to_csv('twibot-22/sample4/subcommunity_sampled_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved sampled human subset \n")

    community_sample_bots.to_csv('twibot-22/sample4/subcommunity_sampled_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved sampled bot subset \n")

if __name__ == "__main__":
    file_paths = glob.glob('twibot-22/preprocessed_twibot22_*.csv') 
    output_path = 'twibot-22/combined_preprocessed_twibot22.csv'
    community_file = 'twibot22/sample4/user1.json'

    combined_df, human_subset, bot_subset = combine_preprocessed_files(file_paths, output_path)
    
    sample_preprocessed_data(human_subset, bot_subset, 1607, 19393)
    #sample_user_community(combined_df, community_file)
