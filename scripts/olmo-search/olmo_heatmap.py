import os
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Helper Functions
# --------------------------

def run_exact_match(correct_author, correct_title_list, returned_author, returned_title, lang):
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
    author_match = fuzz.ratio(unidecode.unidecode(correct_author).lower(), unidecode.unidecode(returned_author).lower()) >= 90 or \
                   unidecode.unidecode(correct_author).lower() in unidecode.unidecode(returned_author).lower()

    both_match = title_match and author_match

    result = {
        f"{lang}_title_match": title_match,
        f"{lang}_author_match": author_match,
        f"{lang}_both_match": both_match
    }
    return result

def extract_title_author(results_column):
    results_column = results_column.fillna('').astype(str).str.strip()
    
    extracted = results_column.str.extract(r'"title":\s*"(.*?)",\s*"author":\s*"(.*?)"')
    
    unmatched_rows = extracted.isnull().all(axis=1)
    extracted.loc[unmatched_rows, 0] = results_column[unmatched_rows]
    extracted.loc[unmatched_rows, 1] = results_column[unmatched_rows]
    
    return extracted

def evaluate(csv_file_name, book_filename, model, prompt_setting, experiment):
    """
    Reads the CSV file, finds the correct book title and author from `book_names.csv`,
    evaluates each result using fuzzy matching, and returns a DataFrame of results.
    """
    book_names = pd.read_csv('scripts/Evaluation/dir_probe/book_names.csv')
    df = pd.read_csv(csv_file_name)
    
    filtered_df = df.loc[:, df.columns.str.contains('results', case=False) & (df.columns != 'Unnamed: 0_results')]
    
    book_title_adjusted = book_filename.replace(f'_{experiment}_{model}_{prompt_setting}.csv', '')
    book_title_adjusted = book_title_adjusted.replace('_', ' ')
    print("Evaluating:", book_title_adjusted)

    if book_title_adjusted == "Alice in Wonderland":
        book_title_adjusted = "Alice s Adventures in Wonderland"
    if book_title_adjusted == "Percy Jackson The Lightning Thief":
        book_title_adjusted = "The Lightning Thief"

    matching_row = book_names[book_names.isin([book_title_adjusted]).any(axis=1)].values.flatten().tolist()
    if not matching_row:
        print(f"No matching book found for title: {book_title_adjusted}")

    author = matching_row[0]
    results_all = pd.DataFrame()

    for column in filtered_df.columns:
        lang_results = []
        filtered_column = extract_title_author(filtered_df[column])
        for i in range(len(filtered_column)):
            returned_title = filtered_column[0].iloc[i]
            returned_author = filtered_column[1].iloc[i]
            eval_result = run_exact_match(author, matching_row, returned_author, returned_title, column)
            lang_results.append(eval_result)
        lang_results_df = pd.DataFrame(lang_results)
        results_all = pd.concat([results_all, lang_results_df], axis=1)

    return results_all

def save_data(title, data, main_dir, subfolder=""):
    """
    Saves the DataFrame to a single CSV file.
    If subfolder is provided, the file is saved to main_dir/evaluation/subfolder.
    Otherwise, it is saved to main_dir/evaluation.
    """
    if subfolder:
        eval_dir = os.path.join(main_dir, "evaluation", subfolder)
    else:
        eval_dir = os.path.join(main_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    filename = f"{title}_eval.csv"
    output_path = os.path.join(eval_dir, filename)
    data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved evaluated data: {output_path}")

def parse_filename(base_name):
    """
    Given a base filename (without the .csv extension), this function extracts:
      - title
      - experiment (one of: "direct_probe_non_NE", "direct_probe_masked", or "direct_probe")
      - model
      - prompt_setting ("one-shot" or "zero-shot")
    
    Assumes the filename follows the format:
      {title}_{experiment}_{model}_{prompt_setting}.csv

    Note: The title may contain underscores.
    """
    experiments = [
                   "direct_probe_non_NE", 
                   "direct_probe_masked", 
                   "direct_probe"
                   ]
    
    experiment = None
    experiment_index = None
    for exp in experiments:
        index = base_name.find(exp)
        if index != -1:
            experiment = exp
            experiment_index = index
            break
    if experiment is None:
        raise ValueError(f"Experiment not found in filename: {base_name}")

    title_part = base_name[:experiment_index]
    if title_part.endswith('_'):
        title_part = title_part[:-1]
    title = title_part.replace('_', ' ').strip()

    remainder = base_name[experiment_index + len(experiment):]
    if remainder.startswith('_'):
        remainder = remainder[1:]
    
    prompt_setting = None
    for ps in ["one-shot", "zero-shot"]:
        if ps in remainder:
            prompt_setting = ps
            break
    if prompt_setting is None:
        raise ValueError(f"Prompt setting ('one-shot' or 'zero-shot') not found in filename: {base_name}")

    parts = remainder.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Unable to parse model and prompt setting from remainder: {remainder}")
    model = parts[0].strip()
    
    return title, experiment, model, prompt_setting

def compute_language_accuracy(evaluated_df):
    """
    Computes average accuracy (in percentage) per language.
    It searches for columns that contain "_results_both_match" and uses the substring before that.
    """
    lang_acc = {}
    for col in evaluated_df.columns:
        if "_results_both_match" in col:
            lang = col.split("_results_both_match")[0]
            acc = evaluated_df[col].astype(int).mean() * 100
            lang_acc.setdefault(lang, []).append(acc)
    for lang in lang_acc:
        lang_acc[lang] = sum(lang_acc[lang]) / len(lang_acc[lang])
    return lang_acc

def create_heatmap(heatmap_dict, main_dir, heatmap_filename):
    if not heatmap_dict:
        print(f"No data for {heatmap_filename} heatmap.")
        return
    heatmap_df = pd.DataFrame(heatmap_dict).T.sort_index()

    # Order columns so that preferred languages come first.
    preferred = ['en', 'es', 'tr', 'vi', 'en_masked', 'es_masked', 'tr_masked', 'vi_masked',
                 'st','yo','tn','ty','mai','mg',
                 'en_shuffled','es_shuffled','tr_shuffled','vi_shuffled',
                 'en_masked_shuffled','es_masked_shuffled','tr_masked_shuffled','vi_masked_shuffled',
                 'st_shuffled','yo_shuffled','tn_shuffled','ty_shuffled','mai_shuffled','mg_shuffled']
    ordered_cols = [lang for lang in preferred if lang in heatmap_df.columns]
    remaining_cols = [col for col in heatmap_df.columns if col not in preferred]
    ordered_cols.extend(remaining_cols)
    heatmap_df = heatmap_df[ordered_cols]

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )
    
    eval_dir = os.path.join(main_dir, "evaluation")
    if heatmap_filename == "2024":
        eval_dir = os.path.join(eval_dir, "2024")
    os.makedirs(eval_dir, exist_ok=True)
    
    csv_path = os.path.join(eval_dir, f"aggregate_data_{heatmap_filename}.csv")
    heatmap_df.to_csv(csv_path, index=True)
    print(f"Saved aggregate data CSV: {csv_path}")
    
    plt.figure(figsize=(18, 12))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap=custom_cmap,
                vmin=0, vmax=100,
                cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size": 15})
    plt.xlabel("Language", fontsize=16)
    plt.ylabel("Book Title", fontsize=16)
    plt.title(f"{heatmap_filename} Aggregate Accuracy Heatmap", fontsize=16)
    
    heatmap_path = os.path.join(eval_dir, f"accuracy_heatmap_{heatmap_filename}.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

# --------------------------
# Main Workflow
# --------------------------

if __name__ == "__main__":
    dp_dir = 'results/direct_probe'
    
    subdirs = [os.path.join(dp_dir, d) for d in os.listdir(dp_dir) if os.path.isdir(os.path.join(dp_dir, d))]
    
    for base_dir in subdirs:
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        for main_dir in subdirs:
            
            print(f"\n=== Processing main_dir: {main_dir} ===")
            heatmap_dict_main = {}
            heatmap_dict_2024 = {}
            
            csv_files = [f for f in os.listdir(main_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                full_path = os.path.join(main_dir, csv_file)
                base_name = csv_file[:-4]
                try:
                    title, experiment, model, prompt_setting = parse_filename(base_name)
                except ValueError as e:
                    print(e)
                    continue
                
                print(f"Processing file: {csv_file}")
                print(f"  -> Title: {title}, Experiment: {experiment}, Model: {model}, Prompt Setting: {prompt_setting}")
                evaluated_df = evaluate(full_path, csv_file, model, prompt_setting, experiment)
                if evaluated_df.empty:
                    print("No evaluation results. Skipping file.")
                    continue

                if any(sub in csv_file for sub in ['Below_Zero', 'Bride', 'First_Lie_Wins', 'Funny_Story', 
                                                    'If_Only_I_Had_Told_Her', 'Just_for_the_Summer', 'Lies_and_Weddings', 
                                                    'The_Ministry_of_Time', 'The_Paradise_Problem', 'You_Like_It_Darker_Stories']):
                    subfolder = "2024"
                    save_data(title.replace(' ', '_'), evaluated_df, main_dir, subfolder=subfolder)
                    heatmap_dict_2024[title] = compute_language_accuracy(evaluated_df)
                else:
                    subfolder = ""
                    save_data(title.replace(' ', '_'), evaluated_df, main_dir, subfolder=subfolder)
                    heatmap_dict_main[title] = compute_language_accuracy(evaluated_df)

            create_heatmap(heatmap_dict_main, main_dir, "")
            create_heatmap(heatmap_dict_2024, main_dir, "2024")
