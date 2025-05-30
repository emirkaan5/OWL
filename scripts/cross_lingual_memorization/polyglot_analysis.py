import pandas as pd
from polyglot.detect import Detector
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob


def analyze_with_polyglot(path2csv, language_columns, visualization_path=None):
    book_name = os.path.basename(path2csv).split('_merged.csv')[0]
    print(f"Processing: {book_name}")
    
    df = pd.read_csv(path2csv)

    def count_english_windows(text):
        words = text.split()
        count_en = 0
        count_unreliable = 0
        for i in range(len(words) - 14):  # 15-word windows
            window = ' '.join(words[i:i + 15])
            try:
                detector = Detector(window, quiet=True)
                if detector.language.code == 'en':
                    count_en += 1
            except Exception as e:
                print(f"Unreliable detection for window: '{window}' - {e}")  # Debugging line
                count_unreliable += 1
        return count_en, count_unreliable

    # Analyze and store results in new columns
    for col in language_columns:
        if col in df.columns:
            new_col_en = f"{col}_polyglot_en"
            new_col_unreliable = f"{col}_polyglot_unreliable"
            results = df[col].dropna().apply(lambda x: count_english_windows(str(x)))
            df[new_col_en] = results.apply(lambda x: x[0])
            df[new_col_unreliable] = results.apply(lambda x: x[1])
            print(f"Ran polyglot on col: {col}, added {new_col_en} and {new_col_unreliable}.")
        else:
            print(f"Column '{col}' does not exist in the CSV and is skipped.")
    df.to_csv(path2csv, index=False)

    # Plotting results: bar plot
    if visualization_path:
        polyglot_counts = {col: (df[f"{col}_polyglot_en"].sum(), df[f"{col}_polyglot_unreliable"].sum()) for col in language_columns if f"{col}_polyglot_en" in df.columns}
        viz_df = pd.DataFrame(list(polyglot_counts.items()), columns=['Column', 'Counts'])
        viz_df[['English Count', 'Unreliable Count']] = pd.DataFrame(viz_df['Counts'].tolist(), index=viz_df.index)
        plt.figure(figsize=(10, 6))
        viz_df.drop(columns='Counts').set_index('Column').plot(kind='bar', stacked=True, colormap='viridis')
        plt.title(f'Polyglot English Detection - {book_name}')
        plt.ylabel('Count of Detections')
        plt.xlabel('Language Columns')
        plt.xticks(rotation=45)
        
        # saving plot
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{book_name}_polyglot.png"))
        plt.close()
        print(f"Visualization saved for {book_name}")


def process_all_files_for_polyglot(folder_path, visualization_folder, language_columns):
    # Ensure visualization folder exists
    os.makedirs(visualization_folder, exist_ok=True)
    
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*_merged.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            analyze_with_polyglot(
                csv_file, 
                language_columns, 
                visualization_path=visualization_folder
            )
            print(f"Completed processing {os.path.basename(csv_file)}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_file)}: {str(e)}")


# Folder containing all the CSV files
input_folder = '/Users/alishasrivastava/Documents/GitHub/BEAM/scripts/cross_lingual_memorization/fixing the dataset/non ne prompts check/checks'
visualization_folder = '/Users/alishasrivastava/Documents/GitHub/BEAM/scripts/cross_lingual_memorization/fixing the dataset/non ne prompts check/checks/visualizations'

# Language columns to check for polyglot analysis
language_columns = ['st', 'yo', 'tn', 'ty', 'mg', 'mai']

# Process all files for polyglot analysis
process_all_files_for_polyglot(input_folder, visualization_folder, language_columns) 