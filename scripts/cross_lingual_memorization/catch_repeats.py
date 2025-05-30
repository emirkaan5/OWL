import pandas as pd
from collections import Counter
import tiktoken
import seaborn as sns
import matplotlib.pyplot as plt

def has_repeated_ngrams_in_column(path2csv, language_columns, n, threshold, encoding_name="o200k_base", visualization_path=None):
    df = pd.read_csv(path2csv)
    encoding = tiktoken.get_encoding(encoding_name)

    def get_token_ids(string): #tokenizing for token ids
        return encoding.encode(string)

    def get_ngram(token_ids, n): #calculating ngrams
        return [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]

    def has_repeated_ngrams(text, n, threshold):
        token_ids = get_token_ids(text)
        ngrams = get_ngram(token_ids, n)
        ngram_counts = Counter(ngrams)
        return any(count >= threshold for count in ngram_counts.values())

    """for col in language_columns:
        new_col = f"{col}_has_repeated_ngram"
        if new_col in df.columns:
            df.drop(columns=[new_col], inplace=True)
            print(f"Deleted existing column: {new_col}")""" #TODO: if this alr exists, edit

    repetition_proportions = []
    for col in language_columns:
        if col in df.columns:
            new_col = f"{col}_has_repeated_ngram"
            df[new_col] = df[col].dropna().apply(
                lambda x: has_repeated_ngrams(str(x), n, threshold)
            )
            proportion = df[new_col].mean()  # Calculate proportion of `True` values
            repetition_proportions.append((col, proportion))
            print(f"Ran col: {col}, added {new_col}.")
        else:
            print(f"ERRORR: col '{col}' does not exist in the CSV and is skipped.")
    df.to_csv(path2csv, index=False)


    #Plotting results: bar plot 
    if visualization_path:
        viz_df = pd.DataFrame(repetition_proportions, columns=['Column', 'Repetition Proportion'])
        plt.figure(figsize=(10, 6))
        sns.barplot(data=viz_df, x='Column', y='Repetition Proportion', palette='flare')
        plt.title('Proportion of Rows with Repeated N-Grams')
        plt.ylabel('Proportion')
        plt.xlabel('Column')
        plt.ylim(0, 1) #from 0 to 1 
        plt.xticks(rotation=45)
        
        # saving plot
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()

path2csv = '/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/The_Picture_of_Dorian_Gray/stored/The_Picture_of_Dorian_Gray.csv'
visualization_path = '/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/The_Picture_of_Dorian_Gray/visualizations/repeated_ngrams.png'
language_columns = ['st', 'yo', 'tn', 'ty', 'mai', 'mg', 'dv'] #TODO: make sure u wanna do these cols

#TODO: declare your n-grams and threshold here
n = 15  # len n-grams considered
threshold = 3  # number of repetitions

has_repeated_ngrams_in_column(path2csv, language_columns, n, threshold, visualization_path=visualization_path)
