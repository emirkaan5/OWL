import os
import pandas as pd
import requests
import time
import textwrap


def search_passages_in_infini_gram(root_folder, index_name, output_file):
    """
    Searches passages from the 'en' column in all CSV files within a directory and its subdirectories using the infini-gram API.

    Parameters:
    - root_folder: str : Path to the root folder containing CSV files.
    - index_name: str : The index to search in the infini-gram API.
    - output_file: str : Path to the output CSV file to store results.

    Returns:
    - None
    """
    results = []

    # Walk through all directories and subdirectories
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv') and ("unmasked_passages" in filename or "non_NE" in filename ):
                file_path = os.path.join(dirpath, filename)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)

                    # Check if 'en' column exists
                  #  if 'en'  in df.columns:
                    for col in df.columns:
                        if "en" not in col:
                        # Iterate over each passage in the 'en' column
                            for passage in df[col].dropna().unique():

                                # Split passage into chunks if it's longer than 1000 characters
                                chunks = textwrap.wrap(passage, width=1000, break_long_words=False, break_on_hyphens=False)

                                for chunk in chunks:
                                    payload = {
                                        'index': index_name,
                                        'query_type': 'count',
                                        'query': chunk
                                    }

                                    # Retry mechanism for 429 errors
                                    for attempt in range(5):
                                        response = requests.post('https://api.infini-gram.io/', json=payload)
                                        if response.status_code == 200:
                                            result = response.json()
                                            # print(result)
                                            # print(result["count"])
                                            count = result.get('count', None)
                                            approx = result.get('approx', None)
                                            print(count, approx)
                                            results.append({
                                                'filename': filename,
                                                'passage': chunk,
                                                'count': count,
                                                'approx': approx
                                            })
                                            break
                                        elif response.status_code == 429:
                                            wait_time = 2 ** attempt
                                            print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                                            time.sleep(wait_time)
                                        else:
                                            print(f"API request failed for chunk: {chunk} (Status code: {response.status_code})")
                                            break

                                    # Small delay to avoid hammering the server
                                    time.sleep(1)
                    else:
                        print(f"Column 'en' not found in {filename}.")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
folder_path = '/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts'  # Replace with the path to your folder containing CSV files
index_name = 'v4_rpj_llama_s4'        # Replace with the appropriate index name
output_file = 'search_results_multilingual.csv'    # Replace with your desired output file path
search_passages_in_infini_gram(folder_path, index_name, output_file)
