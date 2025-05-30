import os
import pandas as pd
import requests
import time
import textwrap
import pprint
import logging

# Configure logging

logging.basicConfig(
    filename='script.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Example usage
logging.info("Script started.")


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


    # Read the CSV file
    df = pd.read_csv(root_folder)
    langs = ['en','vi','es','tr']
    # Check if 'en' column exists
    #  if 'en'  in df.columns:
    for col in langs:
        # Keep index while dropping NaN
        non_na_passages = df[col].dropna()

        for idx, passage in non_na_passages.items():
            # Split passage into chunks if it's longer than 1000 characters
            chunks = textwrap.wrap(passage, width=1000, break_long_words=False, break_on_hyphens=False)

            for chunk in chunks:
                payload = {
                    'index': index_name,
                    'query_type': 'infgram_prob',
                    'query': chunk
                }

                for attempt in range(5):
                    response = requests.post('https://api.infini-gram.io/', json=payload)
                    # print(response.status_code)
                    if response.status_code == 200:
                        result = response.json()
                        # count =result.get('count',0)
                        logging.info(result)
                        # if count > 0:

                        # print(result)
                        logging.info("sent payload")
                        result_entry = {
                            "row_index": idx,
                            "original_passage": passage,
                            "chunk": chunk,
                            "count": result.get("prob", 0),
                            "prompt_cnt": result.get("prompt_cnt", 0),
                            "cont_cnt": result.get("cnt_cnt", 0)
                        }
                        results.append(result_entry)

                        # Log the dictionary
                        print("Appended result:\n%s", pprint.pformat(result_entry))
                        for handler in logging.getLogger().handlers:
                            handler.flush()
                        break
                    elif response.status_code == 429:
                        logging.info("RATELIMITTTT")
                        time.sleep(1)
                    else:
                        logging.info(f"Error: {response.status_code} - {response.text}")
                        break

# Optional: convert to DataFrame


                    # Small delay to avoid hammering the server
                    time.sleep(0.1)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")


# Example usage
if __name__ == '__main__':
    folder_path = '/Users/emir/Projects/BEAM/final dataset/non_ne.csv'  # Replace with the path to your folder containing CSV files
    index_name = 'v4_rpj_llama_s4'        # Replace with the appropriate index name
    output_file = 'olmo_search_results_multilingual_non_ne_infinigram.csv'    # Replace with your desired output file path
    search_passages_in_infini_gram(folder_path, index_name, output_file)
