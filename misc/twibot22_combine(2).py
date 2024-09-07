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
            combined_df[['inf', 'type1', 'type2', 'type3']] = 0  # Resetting the columns to be summed
        else:
            # Verify the structure
            if not all(combined_df.columns == df.columns):
                raise ValueError(f"Column mismatch found in file {file_path}")
        
        # Sum the columns that need to be aggregated
        combined_df['inf'] += df['inf']
        combined_df['type1'] += df['type1']
        combined_df['type2'] += df['type2']
        combined_df['type3'] += df['type3']
    
    print(f'Sample of raw combined dataframe: {combined_df.head()}')

    # Convert label column to numerical
    combined_df['label'] = combined_df['label'].apply(lambda x: 1 if x == 'bot' else 0)
    print("Converted labels to numerical values\n")
    
    # Normalize influence scores
    max_inf = combined_df['inf'].max()
    if max_inf > 0:
        combined_df['inf'] = combined_df['inf'] / max_inf
    print("Normalized influence scores\n")
    
    # Normalize tweet type counts
    type_sum = combined_df[['type1', 'type2', 'type3']].sum(axis=1)
    combined_df[['type1', 'type2', 'type3']] = combined_df[['type1', 'type2', 'type3']].div(type_sum, axis=0)
    print("Normalized tweet type counts\n")

    human_subset = combined_df[combined_df['label'] == 0]
    bot_subset = combined_df[combined_df['label'] == 1]
    print("Split the dataset\n")
    
    # Save the combined DataFrame to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Saved combined data to {output_path}")

    human_subset.to_csv('combined_preprocessed_twibot22_humans.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)
    bot_subset.to_csv('combined_preprocessed_twibot22_bots.csv', columns=['userid', 'label', 'ff', 'inf', 'type1', 'type2', 'type3'], index=False)

if __name__ == "__main__":
    file_paths = glob.glob('output/preprocessed_twibot22_*.csv')  # Replace with the actual path
    output_path = 'output/combined_preprocessed_twibot22.csv'  # Replace with the desired output path
    combine_preprocessed_files(file_paths, output_path)
