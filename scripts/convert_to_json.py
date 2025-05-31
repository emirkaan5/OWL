import csv
import json
import pandas as pd

def csv_to_json_pandas(csv_filepath, json_filepath):
    """
    Converts a CSV file to a JSON file using Pandas.

    Args:
        csv_filepath: The path to the input CSV file.
        json_filepath: The path to the output JSON file.
    """
    try:
        df = pd.read_csv(csv_filepath, encoding='utf-8')
        df.to_json(json_filepath, orient='records', indent=4)
        print(f"Successfully converted '{csv_filepath}' to '{json_filepath}' using Pandas")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example Usage:
if __name__ == "__main__":
  # Replace with your actual filepaths
  csv_file = '/Users/emir/Projects/BEAM_FINAL/BEAM/final dataset/masked.csv'
  json_file = '/Users/emir/Projects/BEAM_FINAL/BEAM/final dataset/json/masked.json'

  # Create a dummy CSV file for testing (optional - remove for your actual data)
#   with open(csv_file, 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Name', 'Age', 'City'])
#     writer.writerow(['Alice', '30', 'New York'])
#     writer.writerow(['Bob', '25', 'London'])
#     writer.writerow(['Charlie', '35', 'Paris'])

  csv_to_json_pandas(csv_file, json_file)
