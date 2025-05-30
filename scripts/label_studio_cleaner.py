import os
import pandas as pd
import random

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        print(str(item))
        item_path = os.path.join(directory, item)
        # Check if it's a file (not a folder)
        if os.path.isfile(item_path) and item.endswith('.csv'):  # Only CSV files
            folder_names.append(item)
    return folder_names

def shuffle_words(prompt):
    words = prompt.split()
    random.shuffle(words)
    return ' '.join(words)

def run_filter(directory, filename):
    csv_file = os.path.join(directory, filename)  # Create the full path to the file
    df = pd.read_csv(csv_file)

    # Assuming the label is stored in a column named 'choices', filter the dataframe
    aligned_label = "Aligned 😊"

    # Filter the rows where the 'sentiment' column contains 'Aligned 😊'
    filtered_df = df[df['sentiment'].apply(lambda x: aligned_label in x if isinstance(x, str) else False)]
    filtered_df = filtered_df[['Single_ent', 'en', 'es', 'tr', 'vi']]
    
    # Shuffle words in the prompt columns
    filtered_df['en_prompts_shuffled'] = filtered_df['en_prompts'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['es_prompts_shuffled'] = filtered_df['es_prompts'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['tr_prompts_shuffled'] = filtered_df['tr_prompts'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['vi_prompts_shuffled'] = filtered_df['vi_prompts'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)

    # Define the output directory and ensure it exists
    output_directory = os.path.join(directory, 'filtered')
    os.makedirs(output_directory, exist_ok=True)

    # Save the filtered dataframe to a new CSV file
    filtered_csv_file = os.path.join(output_directory, f'{filename}_filtered.csv')
    filtered_df.to_csv(filtered_csv_file, index=False)
    print(f'Filtered CSV saved to {filtered_csv_file}')

if __name__ == "__main__":
    directory = 'label_Studio/'
    print(f"Current Working Directory: {os.getcwd()}")
    titles = get_folder_names(directory)

    for title in titles:
        print(f'----------------- Running {title} -----------------')
        run_filter(directory, title)
