# UnDBot Reimplementation
The codes for ***Unsupervised Social Bot Detection via Structural Information Theory***.

## Directories
botwiki-2019/: Contains preprocessed data specific to the Botwiki-2019 dataset

cresci-2015/: Contains preprocessed data specific to the Cresci-2013 dataset

cresci-2017/: Contains preprocessed data specific to the Cresci-2017 dataset

pronbots-2019/: Contains preprocessed data specific to the Pronbots-2019 dataset

silearn/: Contains the implementation of the structural entropy learning algorithm used by UnDBot (further contents not lsited)

figs/: Contains the figures generated for the Twibot-22 tests and UnDBot performance which are present in the paper

twibot22_preprocess/: Contains the combined preprocessed version of the full Twibot-22 dataset
    
    sample1/: Contains the Twibot-22 sample neccessarry to conduct Experiment 1, Experiment 2, Experiment 5, and Experiment 6
    
    sample2/: Contains the Twibot-22 sample neccessarry to conduct Experiment 3 and Experiment 7
    
    sample3/: Contains the Twibot-22 sample neccessarry to conduct Experiment 4
    
    sample4/: Contains the Twibot-22 sample neccessarry to conduct Experiment 8

misc/: Contains a sample of various testing files and drafts of scripts which were created throughout the process of reimplementation

### Relevant Twibot-22 Files
twibot-22_preprocess.py: The raw preprocessing script for the Twibot-22 dataset, which takes as input the raw dataset (not included), and produces as output 9 raw preprocessed files (not included)

twibot-22_combine.py: The script for combining the 9 raw preprocessed output files into one dataframe, then applying any further normalization and further preprocessing (such as sampling) on the combined dataframe. It takes as input the 9 raw preprocessed output files, and generates as output a final combined preprocessed file, along with one file for each human and bot subset in the combined preproccesed file. If the sampling function is used, it further outputs a combined preprocessed sample, and one file for each human and bot subset of the combined preproccessed sample.

twibot-22_main.py: The main file for implementing UnDBot on the Twibot-22 data. It takes as input the human and bot subets of the desired Twibot-22 sample, and produces debugging output along with performance scores and an ROC curve figure.

twibot-22_graphs.py: Generates the figures for analysing the statistical distribution of a Twibot-22 sample. Takes as input the desired Twibot-22 sample and produces as output a series of figures (displayed in the paper)

cresci15.py: Script for processing and analyzing the Cresci-2015 dataset

cresci17.py: Script for processing and analyzing the Cresci-2017 dataset

main.py: Script for processing and analyzing the Botwiki-2019 dataset

util.py: Utility functions used throughout the project for tasks such as data manipulation, feature extraction, or performance evaluation, including implementations of the MultiRank algorithm 

#### Requirements
The implementation of UnDBot is under Python 3.10.6, full package requirements are present in the 'requirements.txt' file
