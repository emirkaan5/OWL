import os
import re
import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import ast

# Extracts text between <output> and </output>
def extract_output_names(text):
    matches = re.findall(r'<output>(.*?)</output>', text)
    return " ".join(matches) if matches else text

# Normalize result strings
def preprocess_result(val):
    if isinstance(val, str):
        processed = extract_output_names(val)
        return unidecode.unidecode(processed).strip().lower()
    return val

# Expand entity mentions to include token-level variants
def granular_ents(val):
    expanded_list = []
    for item in val:
        if isinstance(item, str):
            expanded_list.append(item)
            expanded_list.extend(item.split())
    return list(set(expanded_list))

# Ensure the ground truth is a list
def ensure_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    raise ValueError(f"Ground truth must be in list form. Got: {val}")

# Calculate exact and fuzzy match scores
def calculate_match_scores(ents_list, pred_result):
    pred_norm = unidecode.unidecode(str(pred_result)).lower().strip()
    exact_match = 0
    highest_fuzzy_score = 0
    for ent in ents_list:
        ent_norm = unidecode.unidecode(ent).lower().strip()
        if ent_norm == pred_norm:
            exact_match = 1
        fuzzy_score = fuzz.ratio(ent_norm, pred_norm) / 100
        highest_fuzzy_score = max(highest_fuzzy_score, fuzzy_score)
    return exact_match, highest_fuzzy_score

# Main evaluation function
def evaluate_predictions(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Ensure Single_ent is a list and expand
    df['Single_ent'] = df['Single_ent'].apply(ensure_list).apply(granular_ents)

    # Identify languages by checking _results columns
    available_langs = [col.split('_results')[0] for col in df.columns if col.endswith('_results')]

    # Preprocess prediction columns
    for col in df.columns:
        if col.endswith('_results') or col.endswith('_shuffled_results'):
            df[col] = df[col].apply(preprocess_result)

    # Evaluate for each language
    for lang in available_langs:
        result_col = f'{lang}_results'
        if result_col in df.columns:
            df[f'{lang}_exact_match'], df[f'{lang}_highest_fuzzy_match'] = zip(*df.apply(
                lambda row: calculate_match_scores(row['Single_ent'], row[result_col]), axis=1))
            df[f'{lang}_correct'] = df.apply(
                lambda row: 'correct' if (
                    row[f'{lang}_exact_match'] == 1 or row[f'{lang}_highest_fuzzy_match'] >= 0.7
                ) else 'incorrect', axis=1)

    # Select columns for output
    output_cols = ['Single_ent']
    for lang in available_langs:
        result_cols = [f'{lang}_results', f'{lang}_exact_match', f'{lang}_highest_fuzzy_match', f'{lang}_correct']
        output_cols.extend([col for col in result_cols if col in df.columns])

    # Save output
    df[output_cols].to_csv(output_csv_path, index=False, encoding='utf-8')


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

files = list_csv_files("EMNLP_results/name_cloze/audio")
files = [f.replace('.csv', '') for f in files]

for f in files:
    evaluate_predictions(
        input_csv_path=f'EMNLP_results/name_cloze/audio/{f}.csv',
        output_csv_path=f'scripts/Evaluation/nct/eval/audio/{f}_eval.csv'
    )
