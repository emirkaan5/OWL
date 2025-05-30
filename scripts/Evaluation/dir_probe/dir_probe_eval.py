import pandas as pd
from fuzzywuzzy import fuzz
import unidecode
import os
import re

def run_fuzzy_match(correct_author, correct_title, returned_author, returned_title):
    correct_author = str(correct_author) if pd.notna(correct_author) else ''
    correct_title = str(correct_title) if pd.notna(correct_title) else ''
    returned_author = str(returned_author) if pd.notna(returned_author) else ''
    returned_title = str(returned_title) if pd.notna(returned_title) else ''
    
    # Normalize
    correct_title = unidecode.unidecode(correct_title).lower()
    returned_title = unidecode.unidecode(returned_title).lower()
    correct_author = unidecode.unidecode(correct_author).lower()
    returned_author = unidecode.unidecode(returned_author).lower()

    # Title match
    title_match = fuzz.ratio(returned_title, correct_title) >= 90 or correct_title in returned_title

    # Author match
    author_match = fuzz.ratio(returned_author, correct_author) >= 90 or correct_author in returned_author

    return title_match, author_match, title_match and author_match

def extract_title_author_field(text):
    """
    Extracts title and author from a string that looks like:
    '{"title": "X", "author": "Y"}'
    """
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    match = re.search(r'"title":\s*"(.*?)",\s*"author":\s*"(.*?)"', text)
    if match:
        return match.group(1), match.group(2)
    return text, text  # fallback: just use raw string for both

def evaluate_csv(input_path: str, output_path: str):

    # ── 1.  add any extra aliases you need here ──────────────────────────
    TITLE_ALIAS_MAP = {
        "1984": ["Nineteen Eighty-Four", "Nineteen Eighty Four"],

    }
    # ─────────────────────────────────────────────────────────────────────

    df = pd.read_csv(input_path)
    output_df = pd.DataFrame(index=df.index)

    # Always keep ground-truth columns for manual inspection
    output_df["en_book_title"] = df["en_book_title"]
    output_df["author_name"]   = df["author_name"]

    languages = ["en", "es", "tr", "vi", "st", "yo", "tn", "ty", "mai", "mg"]
    variants  = ["results", "shuffled_results"]

    for lang in languages:
        title_col          = f"{lang}_book_title"
        fallback_title_col = "en_book_title"

        # Build a per-row list of ground-truth titles (including aliases)
        if title_col in df.columns:
            correct_title_pairs = list(zip(df[title_col], df[fallback_title_col]))
        else:                                 # language has no own-title column
            correct_title_pairs = [(t, t) for t in df[fallback_title_col]]

        correct_authors = df["author_name"]

        for variant in variants:
            pred_col = f"{lang}_{variant}"
            if pred_col not in df.columns:
                continue  # nothing to evaluate for this (lang, variant)

            # keep raw predictions for reference
            output_df[pred_col] = df[pred_col]

            title_match_flags  = []
            author_match_flags = []
            both_match_flags   = []

            for i in range(len(df)):
                pred_text        = df.at[i, pred_col]
                pred_title, pred_author = extract_title_author_field(pred_text)

                # list of candidate ground-truth titles for this row
                title_1, title_2 = correct_title_pairs[i]
                en_title         = df.at[i, "en_book_title"]
                alias_list       = TITLE_ALIAS_MAP.get(str(en_title), [])

                candidate_titles = [title_1]
                if title_2 and title_2 != title_1:
                    candidate_titles.append(title_2)
                candidate_titles.extend(alias_list)

                # ── evaluate title & author ─────────────────────────────
                # author match is independent of the title candidate
                _, author_match, _ = run_fuzzy_match(
                    correct_authors[i], candidate_titles[0],  # any correct title
                    pred_author, pred_title
                )

                # title match: true if *any* alias passes
                title_match = False
                for cand in candidate_titles:
                    tm, _, _ = run_fuzzy_match(
                        correct_authors[i], cand,
                        pred_author, pred_title
                    )
                    if tm:
                        title_match = True
                        break

                both_match = title_match and author_match

                title_match_flags.append(title_match)
                author_match_flags.append(author_match)
                both_match_flags.append(both_match)

            # write boolean evaluation columns
            output_df[f"{lang}_{variant}_title_match"]  = title_match_flags
            output_df[f"{lang}_{variant}_author_match"] = author_match_flags
            output_df[f"{lang}_{variant}_both_match"]   = both_match_flags

    output_df.to_csv(output_path, index=False)
    print(f"Saved evaluation results to {output_path}")
    return output_df


def list_csv_files(folder_path, recursive=False):
    csv_files = []
    
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(file)
    else:
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                csv_files.append(file)
    
    return csv_files

if __name__ == "__main__":
    variants = ['unmasked', 'masked', 'non_ne']
    
    for v in variants:
        files = list_csv_files(f"EMNLP_results/direct_probe/{v}/audio")
        files = [f.replace('.csv', '') for f in files]

        for f in files:
            evaluate_csv(
                f"EMNLP_results/direct_probe/{v}/audio/{f}.csv",
                f"scripts/Evaluation/dir_probe/eval/audio/{v}/{f}_eval.csv"
            )
    # print(extract_title_author_field('<output>"title": "The Picture of Dorian Gray","author": "Oscar Wilde"</output>'))