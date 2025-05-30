import pandas as pd

def calculate_true_ratio(csv_path):
    try:
        df = pd.read_csv(csv_path)

        if 'en_results_both_match' not in df.columns:
            print(f"Column 'en_results_both_match' not found in {csv_path}")
            return

        total_rows = df['en_results_both_match'].notna().sum()
        true_count = df['en_results_both_match'].sum()  

        ratio = (true_count / total_rows) * 100 if total_rows > 0 else 0
        print(f"Ratio of True in 'en_results_both_match': {ratio:.4f} ({true_count}/{total_rows})")

    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")

# Example usage
calculate_true_ratio("scripts/Evaluation/dir_probe/eval/text/unmasked/direct_probe_Qwen2.5-7B-Instruct-1M_one-shot_eval.csv")
