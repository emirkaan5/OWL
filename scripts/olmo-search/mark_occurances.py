
import sys
import pandas as pd
import sys
import pandas as pd
from rapidfuzz import process, fuzz

def merge_side_by_side(file1, file2):
    """
    Load and merge two CSVs side by side (column-wise).
    Assumes rows align correctly by index.
    """
    df1 = pd.read_csv(file1).reset_index(drop=True)
    df2 = pd.read_csv(file2).reset_index(drop=True)
    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df

def mark_occured_from_df(df_merged, search_file, output_file, threshold=85):
    """
    Add 'occured' column to merged DataFrame based on exact or fuzzy matching with search_file.
    """
    df_search = pd.read_csv(search_file)
    df_search_filtered = df_search[df_search["chunk"].str.split().str.len() >= 5]
    valid_passages = list(df_search_filtered["original_passage"].dropna().unique())

    language_columns = ["en", "es", "vi", "tr"]

    def match_passage(p):
        if not isinstance(p, str) or len(p.split()) < 5:
            return False
        if p in valid_passages:
            return True
        match, score, _ = process.extractOne(p, valid_passages, scorer=fuzz.ratio)
        return score >= threshold

    df_merged["occured"] = df_merged[language_columns].apply(
        lambda row: any(match_passage(p) for p in row),
        axis=1
    )
        # Print match stats
    true_count = df_merged["occured"].sum()
    false_count = len(df_merged) - true_count
    total = len(df_merged)
    print(f"Matched rows:   {true_count} ({true_count/total:.2%})")
    print(f"Unmatched rows: {false_count} ({false_count/total:.2%})")

    df_merged.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
if __name__ == "__main__":
    path = "/Users/emir/Projects/BEAM/olmo-search/name_cloze_OLMo-2-1124-13B-Instruct_one-shot_eval.csv"
    dataset_final = "/Users/emir/Projects/BEAM/final dataset/masked.csv"
    merged_df = merge_side_by_side(path, dataset_final)
    mark_occured_from_df(merged_df, sys.argv[1], sys.argv[2])
