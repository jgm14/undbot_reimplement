import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#Dataset has columns 'label', 'ff', 'inf', 'type1', 'type2', 'type3'
#'label': 1 for bots, 0 for humans
#'ff': follow-to-follower ratio
#'inf': posting influence score
#'type1', 'type2', 'type3': posting type distribution

#Follow-to-Follower Statistics with Log Scale
def ff_graph(data, output_path):
    plt.figure(figsize=(10, 10))
    sns.histplot(data[data['label'] == 1]['ff'], color='r', label='bot', kde=True, stat="density", bins=30, log_scale=(True, False), element="step", fill=True, alpha=0.6)
    sns.histplot(data[data['label'] == 0]['ff'], color='b', label='human', kde=True, stat="density", bins=30, log_scale=(True, False), element="step", fill=True, alpha=0.6)
    sns.rugplot(data[data['label'] == 1]['ff'], color='r', height=0.05)
    sns.rugplot(data[data['label'] == 0]['ff'], color='b', height=0.05)
    plt.title('Follow-to-Follower Ratio Distribution (Log Scale)', fontsize=20)
    plt.xlabel('Follow-to-Follower Ratio (Log Scale)', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.xticks(fontsize=18)  # Adjust the fontsize value as needed
    plt.yticks(fontsize=18)  # Adjust the fontsize value as needed
    plt.figtext(0.5, -0.1, 'This is a caption for the figure.', ha='center', fontsize=16)
    # Save figure
    plt.savefig(output_path + 'ff_graph', dpi=300)
    plt.close()

# Posting Type Distribution Statistics
def types_graph(data, output_path):

    #Create bins for retweet and original tweet ratios
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1

    data['retweet_bin'] = pd.cut(data['type2'], bins, include_lowest=True)
    data['original_tweet_bin'] = pd.cut(data['type1'], bins, include_lowest=True)

    # For bots
    bot_data = data[data['label'] == 1]
    bot_heatmap_data = pd.pivot_table(bot_data, values='type3', index='original_tweet_bin', columns='retweet_bin', aggfunc='count').fillna(0)

    plt.figure(figsize=(10, 10))
    sns.heatmap(bot_heatmap_data, annot=True, cmap='Blues', cbar_kws={'label': 'Count'}, annot_kws={"size": 14}, fmt=".0f")
    plt.title('Posting Type Distribution for Bots', fontsize=22)
    plt.xlabel('Retweet Ratio', fontsize=16)
    plt.ylabel('Original Tweet Ratio', fontsize=18)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.figtext(0.5, -0.1, 'This is a caption for the figure.', ha='center', fontsize=16)
    plt.savefig(output_path + 'types_graph_bots', dpi=300)
    plt.close()

    # For humans
    human_data = data[data['label'] == 0]
    human_heatmap_data = pd.pivot_table(human_data, values='type3', index='original_tweet_bin', columns='retweet_bin', aggfunc='count').fillna(0)

    plt.figure(figsize=(10, 10))
    sns.heatmap(human_heatmap_data, annot=True, cmap='Blues', cbar_kws={'label': 'Count'}, annot_kws={"size": 14}, fmt=".0f")
    plt.title('Posting Type Distribution for Humans', fontsize=22)
    plt.xlabel('Retweet Ratio', fontsize=16)
    plt.ylabel('Original Tweet Ratio', fontsize=18)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.figtext(0.5, -0.1, 'This is a caption for the figure.', ha='center', fontsize=16)
    # Save figure
    plt.savefig(output_path + 'types_graph_humans', dpi=300)
    plt.close()

def inf_graph(data, output_path):
    # Create bins for the influence scores
    bins = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

    data['inf_bin'] = pd.cut(data['inf'], bins, include_lowest=True)

    # Calculate the proportion of bots, humans, and misjudged cases in each bin
    proportion_data = data.groupby(['inf_bin', 'label']).size().unstack(fill_value=0)
    proportion_data = proportion_data.div(proportion_data.sum(axis=1), axis=0) * 100

    # Rename columns for clarity
    proportion_data.columns = ['human', 'bot']

    # Plotting
    plt.figure(figsize=(10, 10))
    proportion_data.plot(kind='bar', stacked=True, color=['blue', 'red', 'gray'])

    # Adding labels
    plt.title('Posting Influence Distribution for Bots and Humans', fontsize=16)
    plt.xlabel('Posting Influence', fontsize=16)
    plt.ylabel('Proportion', fontsize=16)
    plt.legend(['human', 'bot', 'misjudge'])
    plt.xticks(rotation=45)
    plt.figtext(0.5, -0.1, 'This is a caption for the figure.', ha='center', fontsize=16)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path + 'inf_graph', dpi=300)
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    print(f'Start time is {time.strftime("%x %X")}')

    # Load sample
    data = pd.read_csv('twibot-22/sample1/sampled_combined_preprocessed_twibot22.csv')
    output_path = 'figs/sample1/'
    ff_graph(data, output_path)
    types_graph(data, output_path)
    inf_graph(data, output_path)
    
    end_time = time.time()
    print(f'running time: {(end_time - start_time) / 1} s')
    print(f'End time is {time.strftime("%x %X")}')
