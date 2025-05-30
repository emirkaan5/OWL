import os
import csv

# Define the base directory
base_dir = '/Users/alishasrivastava/BEAM/scripts/Prompts'
output_file = '/Users/alishasrivastava/BEAM/filled_cells_count.txt'

# List of directories to exclude
exclude_dirs = ['2024', 'Alice_in_Wonderland']

# Columns to check for {lang}_gt
lang_gt_columns = ['st_gt', 'yo_gt', 'tn_gt', 'ty_gt', 'mai_gt', 'mg_gt']

# Columns to check for language codes
lang_columns = ['en_masked', 'es_masked', 'tr_masked', 'vi_masked']

# Initialize a dictionary to store counts for {lang}_gt
counts_gt = {col: 0 for col in lang_gt_columns}

# Initialize a dictionary to store total counts for {lang}_gt
row_counts_gt = {col: 0 for col in lang_gt_columns}

# Initialize a dictionary to store counts for language codes
counts_lang = {col: 0 for col in lang_columns}

# Initialize a dictionary to store total counts for language codes
row_counts_lang = {col: 0 for col in lang_columns}

def count_filled_cells_in_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col in lang_gt_columns:
                if col in row:
                    row_counts_gt[col] += 1
                    if row[col].strip():
                        counts_gt[col] += 1
            for col in lang_columns:
                if col in row:
                    row_counts_lang[col] += 1
                    if row[col].strip():
                        counts_lang[col] += 1

# Traverse through each directory in the base directory
for folder_name in os.listdir(base_dir):
    if folder_name in exclude_dirs or not os.path.isdir(os.path.join(base_dir, folder_name)):
        continue
    
    stored_dir = os.path.join(base_dir, folder_name, 'stored')
    csv_file = os.path.join(stored_dir, f'{folder_name}.csv')
    
    if os.path.exists(csv_file):
        count_filled_cells_in_csv(csv_file)

# Calculate the total number of filled cells and total possible cells for {lang}_gt
total_filled_cells_gt = sum(counts_gt.values())
total_possible_cells_gt = sum(row_counts_gt.values())

# Calculate the total number of filled cells and total possible cells for language codes
total_filled_cells_lang = sum(counts_lang.values())
total_possible_cells_lang = sum(row_counts_lang.values())

# Calculate the total number of unfilled cells for {lang}_gt
total_unfilled_cells_gt = total_possible_cells_gt - total_filled_cells_gt

# Calculate the total number of possible cells across all columns
total_possible_cells_all = total_possible_cells_gt + total_possible_cells_lang

# Write the results to the output file
with open(output_file, 'w') as f:
    f.write('Counts for {lang}_gt columns:\n')
    for col in lang_gt_columns:
        percent_filled = (counts_gt[col] / row_counts_gt[col] * 100) if row_counts_gt[col] > 0 else 0
        f.write(f'{col}: {counts_gt[col]} out of {row_counts_gt[col]} ({percent_filled:.2f}%)\n')
    total_percent_filled_gt = (total_filled_cells_gt / total_possible_cells_gt * 100) if total_possible_cells_gt > 0 else 0
    total_percent_unfilled_gt = (total_unfilled_cells_gt / total_possible_cells_gt * 100) if total_possible_cells_gt > 0 else 0
    f.write(f'Total filled cells: {total_filled_cells_gt} out of {total_possible_cells_gt} ({total_percent_filled_gt:.2f}%)\n')
    f.write(f'Total unfilled cells: {total_unfilled_cells_gt} out of {total_possible_cells_gt} ({total_percent_unfilled_gt:.2f}%)\n\n')

    f.write('Counts for language columns (en_masked, es_masked, tr_masked, vi_masked):\n')
    for col in lang_columns:
        percent_filled = (counts_lang[col] / row_counts_lang[col] * 100) if row_counts_lang[col] > 0 else 0
        f.write(f'{col}: {counts_lang[col]} out of {row_counts_lang[col]} ({percent_filled:.2f}%)\n')
    total_percent_filled_lang = (total_filled_cells_lang / total_possible_cells_lang * 100) if total_possible_cells_lang > 0 else 0
    f.write(f'Total filled cells for language columns: {total_filled_cells_lang} out of {total_possible_cells_lang} ({total_percent_filled_lang:.2f}%)\n\n')

    f.write(f'Total possible cells across all columns: {total_possible_cells_all}\n')

print(f'Results written to {output_file}') 