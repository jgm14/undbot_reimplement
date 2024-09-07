import pandas as pd
import glob

def combine_preprocessed_files(file_paths, output_path):
    combined_df = None
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        
        if combined_df is None:
            # Initialize combined_df with the first file, except for the columns to be summed
            combined_df = df.copy()
            combined_df[['inf', 'type1', 'type2', 'type3', 'original_tweet_count']] = 0  # Resetting the columns to be summed
        else:
            # Verify the structure
            if not all(combined_df.columns == df.columns):
                raise ValueError(f"Column mismatch found in file {file_path}")
        
        # Sum the columns that need to be aggregated
        combined_df['inf'] += df['inf'].fillna(0)
        combined_df['type1'] += df['type1']
        combined_df['type2'] += df['type2']
        combined_df['type3'] += df['type3']
        combined_df['original_tweet_count'] += df['original_tweet_count']
    
    print(f'Sample of raw combined dataframe: {combined_df.head()}')

    # Convert label column to numerical
    combined_df['label'] = combined_df['label'].apply(lambda x: 1 if x == 'bot' else 0)
    print("Converted labels to numerical values\n")
    
    # Calculate average influence scores
    combined_df['inf'] = combined_df.apply(lambda row: row['inf'] / row['original_tweet_count'] if row['original_tweet_count'] > 0 else 0, axis=1)
    #combined_df['inf'] = combined_df.apply(lambda row: (
    #(row['public_metrics_comment_count'] / row['original_tweet_count']) if row['original_tweet_count'] > 0 else 0 +
    #(row['public_metrics_like_count'] / row['original_tweet_count']) if row['original_tweet_count'] > 0 else 0 +
    #(row['public_metrics_retweet_count'] / row['original_tweet_count']) if row['original_tweet_count'] > 0 else 0
    #), axis=1)
    
    # Normalize influence scores
    max_inf = combined_df['inf'].max()
    if max_inf > 0:
        combined_df['inf'] = combined_df['inf'] / max_inf
    print("Normalized influence scores\n")
    
    # Normalize tweet type counts
    type_sum = combined_df[['type1', 'type2', 'type3']].sum(axis=1)
    combined_df[['type1', 'type2', 'type3']] = combined_df[['type1', 'type2', 'type3']].div(type_sum, axis=0).fillna(0)
    print("Normalized tweet type counts\n")

    human_subset = combined_df[combined_df['label'] == 0]
    bot_subset = combined_df[combined_df['label'] == 1]
    print("Split the dataset into human and bot subsets\n")
    
    # Save the combined DataFrame to CSV
    combined_df.to_csv(output_path, columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved full combined data to {output_path} with {len(combined_df)} records'\n")

    human_subset.to_csv('try/combined_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved full human subset \n")
    bot_subset.to_csv('try/combined_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    print(f"Saved full bot subset \n")

    return human_subset,bot_subset

def sample_preprocessed_data(human_subset, bot_subset, human_sample_size, bot_sample_size):
        # Perform sampling to get 16,340 humans and 2,660 bots
        sampled_humans = human_subset.sample(n=human_sample_size, random_state=1)
        sampled_bots = bot_subset.sample(n=bot_sample_size, random_state=1)
        print('Sampling the human and bot subsets \n')

        # Combine the sampled data
        sampled_data = pd.concat([sampled_humans, sampled_bots])

        # Save the combined sampled data to a new file
        sampled_data.to_csv('try/test1/sampled_combined_preprocessed_twibot22.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
        print(f"Saved sampled combined data \n")

        # Optionally, save the individual sampled human and bot data
        sampled_humans.to_csv('try/test1/sampled_combined_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
        print(f"Saved sampled human subset \n")
        sampled_bots.to_csv('test/test1/sampled_combined_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'],index=False)
        print(f"Saved sampled bot subset \n")

if __name__ == "__main__":
    file_paths = glob.glob('preprocess_final/preprocessed_twibot22_*.csv')  # Replace with the actual path
    output_path = 'try/combined_preprocessed_twibot22.csv'  # Replace with the desired output path
    human_subset, bot_subset = combine_preprocessed_files(file_paths, output_path)
    sample_preprocessed_data(human_subset, bot_subset, 18920, 3080)
