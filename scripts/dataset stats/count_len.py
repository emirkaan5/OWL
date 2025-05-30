import os
import pandas as pd

def sum_unmasked_and_non_ne_rows(root_dir):
    skip_dirs = ['stored', 'visualisations', '2024']
    total_unmasked = 0
    total_non_ne = 0
    num_masked= 0
    num_other = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if any(skip in dirpath for skip in skip_dirs):
            continue

        for file in filenames:
            if not file.endswith('.csv'):
                continue

            file_path = os.path.join(dirpath, file)
            try:
                df = pd.read_csv(file_path)
                num_rows = len(df)
                if "Paper_Towns" in file:
                    continue
                if "_masked_passages" in file:
                    total_unmasked += num_rows
                    num_masked +=1
                    print(f"{file_path} → unmasked: {num_rows} rows")
                elif "non_NE" in file:
                    num_other +=1
                    total_non_ne += num_rows
                    print(f"{file_path} → non_NE: {num_rows} rows")

            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

    print("\n=== Total Rows Summary ===")
    print(f"unmasked_passages total: {total_unmasked}")
    print(f"non_NE total: {total_non_ne}")
    print(f"combined total: {total_unmasked + total_non_ne}")
    print(num_masked,num_other)
    return total_unmasked, total_non_ne

# Example usage
directory_path = '/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts'
sum_unmasked_and_non_ne_rows(directory_path)
