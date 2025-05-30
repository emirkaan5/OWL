import os
import pandas as pd

def compute_language_counts(evaluated_df):
    """
    Returns a dictionary of {language: (total_correct, total_attempts)}
    based on columns ending in "_results_both_match".
    Each row in those columns should contain the string "correct" or "incorrect".
    """
    lang_counts = {}
    for col in evaluated_df.columns:
        if "_correct" in col:
            # e.g. "en_results_both_match" => lang = "en"
            lang = col.split("_correct")[0]
            
            # Convert 'correct' => 1, anything else => 0
            correct_series = (evaluated_df[col].astype(str).str.lower() == "correct").astype(int)
            total_correct = correct_series.sum()
            total_attempts = len(correct_series)

            # Accumulate counts
            if lang not in lang_counts:
                lang_counts[lang] = (0, 0)
            prev_correct, prev_attempts = lang_counts[lang]
            lang_counts[lang] = (
                prev_correct + total_correct,
                prev_attempts + total_attempts
            )
    return lang_counts

def debug_evaluation_folder(eval_dir):
    """
    Reads all *_eval.csv files in `eval_dir`:
      - Computes (total_correct, total_attempts) for each file and prints them.
      - Accumulates a grand total across all files to compute a final overall accuracy.
    """
    # Gather all CSVs that end with _eval.csv
    eval_files = [
        os.path.join(eval_dir, f)
        for f in os.listdir(eval_dir)
        if f.endswith('.csv')
    ]

    if not eval_files:
        print(f"No *_eval.csv files found in {eval_dir}")
        return

    print(f"\nDebugging evaluation folder: {eval_dir}")
    
    # Grand totals across ALL CSVs in this folder
    overall_counts = {}

    # Process each CSV
    for csv_file in eval_files:
        print(f"\nFile: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue

        lang_counts = compute_language_counts(df)
        if not lang_counts:
            print("  No matching '_results_both_match' columns found.")
            continue

        # Print per-file details
        print("  Per-file language breakdown:")
        for lang, (corr, att) in lang_counts.items():
            if att == 0:
                print(f"    {lang}: 0/0 = 0.00%")
            else:
                accuracy = corr / att * 100
                print(f"    {lang}: {corr}/{att} = {accuracy:.2f}%")

        # Accumulate into overall_counts
        for lang, (corr, att) in lang_counts.items():
            if lang not in overall_counts:
                overall_counts[lang] = (0, 0)
            prev_corr, prev_att = overall_counts[lang]
            overall_counts[lang] = (prev_corr + corr, prev_att + att)

    # Print overall results
    if overall_counts:
        print("\nOverall counts across ALL *_eval.csv in this folder (per language):")
        total_corr_sum = 0
        total_att_sum = 0
        for lang, (corr, att) in overall_counts.items():
            total_corr_sum += corr
            total_att_sum += att
            if att == 0:
                print(f"  {lang}: 0/0 = 0.00%")
            else:
                accuracy = corr / att * 100
                print(f"  {lang}: {corr}/{att} = {accuracy:.2f}%")

        if total_att_sum == 0:
            overall_accuracy = 0.0
        else:
            overall_accuracy = total_corr_sum / total_att_sum * 100
        print(f"\nGrand total (all languages combined): {total_corr_sum}/{total_att_sum} = {overall_accuracy:.2f}%")
    else:
        print("No language data found in this folder.")

if __name__ == "__main__":
    # Example usage:
    # Replace this path with a valid path to your 'evaluation/' folder
    evaluation_folder_path = "results/name_cloze copy/gpt-4o-2024-11-20/one-shot/evaluation"
    debug_evaluation_folder(evaluation_folder_path)
