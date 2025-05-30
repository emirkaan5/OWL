import os
import csv
import re
from collections import defaultdict

def count_words_group_by_lang(root_dir, output_csv="grouped_word_counts.csv"):
    lang_suffixes = ['en', 'es', 'tr', 'vi']
    grouped_counts = defaultdict(dict)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                match = re.match(r"(.+?)_(" + "|".join(lang_suffixes) + r")_processed\.txt", file)
                if match:
                    book_name, lang = match.groups()
                    file_path = os.path.join(dirpath, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            word_count = len(text.split())
                            grouped_counts[book_name][lang] = word_count
                    except Exception as e:
                        print(f"Failed to read {file_path}: {e}")
                        grouped_counts[book_name][lang] = None  # Keep track of missing file

    # Write to CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Book Name", "en", "es", "tr", "vi"]
        writer.writerow(header)
        for book in sorted(grouped_counts.keys()):
            row = [book]
            for lang in lang_suffixes:
                row.append(grouped_counts[book].get(lang, ""))
            writer.writerow(row)

    print(f"\nSaved grouped word counts to {output_csv}")
    return grouped_counts

# Example usage:
directory_path = "/home/ekorukluoglu_umass_edu/beam2/BEAM/alignment/preprocess_books/processed"
count_words_group_by_lang(directory_path)
