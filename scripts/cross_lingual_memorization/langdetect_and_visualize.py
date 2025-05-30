import pandas as pd
from langdetect import detect
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob


def detect_language_in_columns(path2csv, language_columns, visualization_path=None):
    book_name = os.path.basename(path2csv).split('_merged.csv')[0]
    print(f"Processing: {book_name}")
    
    df = pd.read_csv(path2csv)

    # Detect language and store in new columns
    for col in language_columns:
        if col in df.columns:
            new_col = f"{col}_langdetect"
            df[new_col] = df[col].dropna().apply(lambda x: detect(str(x)))
            print(f"Ran langdetect on col: {col}, added {new_col}.")
        else:
            print(f"Column '{col}' does not exist in the CSV and is skipped.")
    df.to_csv(path2csv, index=False)

    # Plotting results: bar plot
    if visualization_path:
        lang_counts = {col: df[f"{col}_langdetect"].value_counts() for col in language_columns if f"{col}_langdetect" in df.columns}
        viz_df = pd.DataFrame(lang_counts).fillna(0).T
        viz_df['en'] = viz_df.get('en', 0)
        viz_df['other'] = viz_df.sum(axis=1) - viz_df['en']
        plt.figure(figsize=(10, 6))
        viz_df[['en', 'other']].plot(kind='bar', stacked=True, colormap='Paired')
        plt.title(f'Language Detection - {book_name}')
        plt.ylabel('Count')
        plt.xlabel('Language Columns')
        plt.xticks(rotation=45)
        
        # saving plot
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{book_name}_langdetect.png"))
        plt.close()
        print(f"Visualization saved for {book_name}")


def process_all_files_for_langdetect(folder_path, visualization_folder, language_columns):
    # Ensure visualization folder exists
    os.makedirs(visualization_folder, exist_ok=True)
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*_merged.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            detect_language_in_columns(
                csv_file, 
                language_columns, 
                visualization_path=visualization_folder
            )
            print(f"Completed processing {os.path.basename(csv_file)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {str(e)}")


# Folder containing all the CSV files
input_folder = '/Users/alishasrivastava/Documents/GitHub/BEAM/scripts/cross_lingual_memorization/fixing the dataset/unmasked prompts check/checks'
visualization_folder = '/Users/alishasrivastava/Documents/GitHub/BEAM/scripts/cross_lingual_memorization/fixing the dataset/unmasked prompts check/checks/visualizations'

# Language columns to check for language detection
language_columns = ['st', 'yo', 'tn', 'ty', 'mg', 'mai']

# Process all files for language detection
process_all_files_for_langdetect(input_folder, visualization_folder, language_columns) 