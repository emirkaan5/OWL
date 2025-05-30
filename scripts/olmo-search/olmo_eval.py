import pandas as pd
import os
import sys
from rapidfuzz import fuzz
import unidecode
import pathlib
import re
def run_exact_match(correct_author, correct_title_list, returned_author, returned_title, lang):
    """
    Purpose: Compare a ground‑truth author/title against a returned 
    author/title for one language using fuzzy matching and substring checks.
    Inputs:
        correct_author – the expected author string
        correct_title_list – list of acceptable title strings
        returned_author – the author extracted from the model output
        returned_title – the title extracted from the model output
        lang – language code (e.g. “en”, “es”)
    Outputs:
        A dict with keys {lang}_title_match, {lang}_author_match, 
        and {lang}_both_match (all booleans).
    """
    returned_author = str(returned_author) if pd.notna(returned_author) else ''
    correct_author = str(correct_author) if pd.notna(correct_author) else ''

    # Check if the returned title matches any of the titles in the correct_title_list using fuzzy matching
    title_match = any(
        fuzz.ratio(unidecode.unidecode(str(returned_title)).lower(), unidecode.unidecode(str(title)).lower()) >= 90
        for title in correct_title_list
    ) or any(
        unidecode.unidecode(str(title)).lower() in unidecode.unidecode(str(returned_title)).lower()
        for title in correct_title_list
    )

    # Check if the authors match using fuzzy matching
    author_match = fuzz.ratio(unidecode.unidecode(correct_author).lower(), unidecode.unidecode(returned_author).lower()) >= 90 or unidecode.unidecode(correct_author).lower() in unidecode.unidecode(returned_author).lower()

    # Check if both title and author match
    both_match = True if title_match == author_match == True else False

    result = {
        f"{lang}_title_match": title_match,
        f"{lang}_author_match": author_match,
        f"{lang}_both_match": both_match
    }
    return result


def extract_title_author(results_column):
    """
    -Purpose: Pull out "title" and "author" fields from a JSON‐style string 
    column using regex. If the pattern fails, it preserves the original 
    cell in both columns.
    -Input: A pandas Series of raw result strings.
    -Output: A two‑column DataFrame: first column with titles, second with authors.
    """
    # Ensure input is cleaned and standardized
    results_column = results_column.fillna('').astype(str).str.strip()
    
    # Extract title and author from the results column using regex
    extracted = results_column.str.extract(r'"title":\s*"(.*?)",\s*"author":\s*"(.*?)"')
    
    # Replace NaN rows with original content
    unmatched_rows = extracted.isnull().all(axis=1)  # Identify rows where regex didn't match
    extracted.loc[unmatched_rows, 0] = results_column[unmatched_rows]
    extracted.loc[unmatched_rows, 1] = results_column[unmatched_rows]
    
    return extracted


def build_file_dict(project_folder):
    """
    Traverse a directory tree using os.walk and build a mapping
    from filename to its full path.
    """
    file_dict = {}
    for dirpath, _, filenames in os.walk(project_folder):
        for fname in filenames:
            file_dict[fname] = os.path.join(dirpath, fname)
    return file_dict

from pathlib import Path

def split_filename(
    filename: str,
    known_suffixes: tuple[str, ...] = ("non_NE", "unmasked_passages")
) -> tuple[str, str]:
    """
    Split *any* BEAM filename into a clean book title and a suffix.

    Returns
    -------
    primary   : str   # Book title with spaces, not underscores
    secondary : str   # The recognised suffix ('' if none)
    """
    stem = Path(filename).stem          # strip the .csv
    # 1.  Look for a *known* suffix from the right‑hand side
    for suffix in known_suffixes:
        tag = "_" + suffix
        if stem.endswith(tag):
            primary = stem[:-len(tag)]
            return primary, suffix

    # 2.  Fallback: treat the last token as the suffix
    primary, _, secondary = stem.rpartition("_")
    if not primary:                      # no underscore at all
        primary, secondary = stem, ""
    return primary, secondary


def find_matching_file(splitted, file_dict, *, use_full_path=True):
    """
    Return a list of file paths whose *stem* (or full path) contains the
    normalised primary title and, if present, the secondary suffix.

    Parameters
    ----------
    splitted : tuple[str, str]
        (primary_title, secondary_suffix)
    file_dict : dict[str, str]
        {filename : full_path}
    use_full_path : bool, default False
        If True, search the entire path; otherwise search only the filename stem.
    """
    primary, secondary = splitted
    primary_norm = primary.lower()
    secondary_norm = secondary.lower() if secondary else None

    matches = []
    for fname, full_path in file_dict.items():
        haystack = (full_path if use_full_path else Path(fname).stem).lower()
        if primary_norm in haystack and (not secondary_norm or secondary_norm in haystack):
            matches.append(full_path)

    return matches
def sliding_window_match(text1, text2, threshold=80):
    """
    Compare two texts using a sliding window approach. The window size is set to the length of the shorter text.
    For each window in the longer text, compute the similarity score with the shorter text using fuzz.ratio.
    If any window's score meets or exceeds the threshold, return True indicating a match.
    """
    text1 = text1.lower()
    text2 = text2.lower()
    # Identify which text is shorter to be the sliding window base.
    if len(text1) < len(text2):
        short_text, long_text = text1, text2
    else:
        short_text, long_text = text2, text1

    window_size = len(short_text)
    if window_size == 0:
        return False  # Avoid division by zero or empty comparisons

    # Slide over the longer text one character at a time.
    for i in range(0, len(long_text) - window_size + 1):
        candidate = long_text[i:i+window_size]
        score = fuzz.ratio(short_text, candidate)
        if score >= threshold:
            return True
    return False
def get_en_results(row, file_data):
    """
    Given a row from the main CSV and a DataFrame loaded from the matched file,
    do an exact match on the passage to retrieve the corresponding en_results.
    """
    passage_to_find = row["passage"]
    match = file_data[file_data["passage"] == passage_to_find]
    if not match.empty:
        return match.iloc[0]["en_results"]
    else:
        return None

# Paths and folder definitions:

def evaluate(input_csv_path,project_folder,bookname_csv):
    book_names = pd.read_csv(bookname_csv)
    result_df = []
    file_dict = build_file_dict(project_folder)

# Filter to include only ne_one_shot files
    file_dict = {
        fname: path for fname, path in file_dict.items()
        if "non_ne_one_shot" in path and path.endswith(".csv")
    }
    df_main = pd.read_csv(input_csv_path)
    df_main["en_results"] = None 
# Iterate over each unique filename in df_main:
    for fname in df_main["filename"].unique():
        splitted = split_filename(fname)
        primary, suffix = splitted
        print(primary)
        # Only continue if suffix is unmasked_passages
        if suffix != "unmasked_passages":
            continue

        # if "1984" in fname:
        if "true":
            # Split the target filename from the main CSV into primary and secondary parts.
            if suffix != "unmasked_passages":
                continue

            primary_title = primary.lower().replace(" ", "_")

            # Match any file in the dictionary that contains the primary title (ignoring suffix) and excludes zero_shot
            matching_file_paths = [
                path for filename, path in file_dict.items()
                if primary_title in filename.lower()
                and "zero_shot" not in path.lower()
            ]

            if not matching_file_paths:
                print(f"Matching file not found for: {fname}")
                print(f"Split name: {primary} | {suffix}")
                continue  # Don't exit, just skip this file

            print(f"Found {len(matching_file_paths)} matching files for the book : {fname}")


            # print(matching_file_paths)
            if matching_file_paths:# print(type(matching_file_paths))
                print(f"Found {len(matching_file_paths)} matching files for the book : {fname}")
                for files in matching_file_paths:
                    splittedfilename = files.split('/')
                    # print(splittedfilename[7])
                    if splittedfilename[7] !="OLMo-2-1124-13B-Instruct":
                        continue
                    if "one_shot" in files:
                        # print(files)

                        # print(splittedfilename[7])
                        # print(len(matching_file_paths))
                        print(f"finding occurences in model : {splittedfilename[7]}")
                        try:
                        # if "true":
                            # Load the CSV file corresponding to the matching file.
                            file_df = pd.read_csv(files)
                            # Update rows in the main CSV that reference this filename.
                            for index,rows in df_main.iterrows():
                                if rows['filename'] == fname:
                                    passage = rows["passage"]
                                    # print(rows['filename'])
                                    for index, rows in file_df.iterrows():
                                        book_title = splitted[0].replace("_", " ")
                                        if book_title == "Alice in Wonderland":
                                            book_title = "Alice s Adventures in Wonderland"
                                        if book_title == "Percy Jackson The Lightning Thief":
                                            book_title = "The Lightning Thief"
                                        matching_row = book_names[book_names.isin([book_title]).any(axis=1)].values.flatten().tolist()
                                        author = matching_row[0]
                                        # print(author)
                                        org_passage = rows["en"]
                                        normalized_org = str(org_passage).lower()
                                        normalized_passage = str(passage).lower()
                                        score = sliding_window_match(normalized_org,normalized_passage)
                                        if score:
                                            rows['index'] = index
                                            rows['filename'] = fname
                                            rows['model'] = splittedfilename[7]
                                            rows['path'] = files
                                            result_df.append(rows)

                                            languages = ['en', 'es', 'tr', 'vi','st','yo','tn','ty','mai','mg']
                                            pattern = re.compile(r'"title":\s*"([^"]+)"\s*,\s*"author":\s*"([^"]+)"')

                                            for lang in languages:
                                                raw_val = rows[f'{lang}_results']

                                                if pd.isna(raw_val):                # <- skip NaNs
                                                    title = author = None
                                                else:
                                                    m = pattern.search(str(raw_val))
                                                    title, author = m.groups() if m else (None, None)
                                                # print(title,author)
                                                # print(extracted)
                                                returned_title = title  # Title is in index 0
                                                returned_author = author  # Author is in index 1
                                                # print(returned_author,returned_title)
                                                # Run exact match evaluation
                                                eval_result = run_exact_match(author, matching_row, returned_author, returned_title,lang)
                                                #print(f"Evaluation result for passage {i}: {eval_result}")
                                                rows[f"{lang}_eval"]= eval_result[f"{lang}_both_match"]

                        except Exception as e:
                            print(f"Error processing file {files}: {e}")
            else:
                print(f"Matching file not found for: {fname}")
                print(splitted)
                print(matching_file_paths)
                print(file_dict)
                sys.exit()
        # print(result_df)

    result_dframe = pd.DataFrame(result_df)
    keyword = "_shuffled"

    df_clean = result_dframe.loc[:, ~pd.Index(result_dframe.columns.astype(str)).str.contains(keyword, case=False, regex=False)]
    print("Columns in df_clean:", df_clean.columns.tolist())

    sorted_df = df_clean.sort_values(by='filename', ascending=True)
    # Save the updated main CSV to a new file.
    output_csv_path = "olmo_eval_non_ne_olmo13b.csv"
    sorted_df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")


if __name__=="__main__":
    input_csv_path = "search-results-(1)-csv.csv"
    project_folder = "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe"
    bookname_csv = "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Evaluation/dir_probe/book_names.csv"
    # Load the main CSV which records occurrences of passages:
    # New column to hold results
    evaluate(input_csv_path=input_csv_path,project_folder=project_folder,bookname_csv=bookname_csv)
    # Build the dictionary mapping filenames to full paths: