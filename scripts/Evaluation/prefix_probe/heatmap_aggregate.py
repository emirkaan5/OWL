import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ===================== USER SETTINGS =====================
ALL_SCORE_TYPES = ["BLEU", "ChrF++", "ROUGE-L"] # "BLEURT"
BASE_DIR = "results/prefix_probe"
LANG_LIST = ["en", "es", "vi", "tr"]
EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and","Paper_Towns",  "Ministry", "Paradise", "Funny_Story"
]

# We'll multiply BLEURT, ROUGE-L by 100 to match BLEU & ChrF++ in [0..100]
SCALE_TO_100 = {"ROUGE-L"}  # set of metrics that need scaling from [0..1] to [0..100]

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)

# ============== 1) Folder Discovery ==============
def find_evaluation_folders(base_dir):
    """
    Look for:
      - one-shot:  eval/csv/one-shot/bleurt
      - zero-shot: eval/csv/zero-shot/bleurt
    Returns a list of tuples: (model_name, folder_path, experiment_type)
    """
    found = []
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # one-shot folder
        one_shot_path = os.path.join(model_path, "eval", "csv", "one-shot", "bleurt")
        if os.path.isdir(one_shot_path):
            found.append((model_name, one_shot_path, "one-shot"))

        # zero-shot folder
        zero_shot_path = os.path.join(model_path, "eval", "csv", "zero-shot", "bleurt")
        if os.path.isdir(zero_shot_path):
            found.append((model_name, zero_shot_path, "zero-shot"))

    return found

# ============== 2) Gather the data for ALL metrics ==============
def load_and_process_data_all_metrics(score_types):
    """
    Collects data for multiple score types (e.g. ["ChrF++", "BLEU", "BLEURT", "ROUGE-L"]).
    
    Returns a nested dict:
      aggregator[exp_type][model][metric][lang] = list of float values
    """
    aggregator = {"one-shot": {}, "zero-shot": {}}
    folders = find_evaluation_folders(BASE_DIR)

    for (model_name, folder_path, exp_type) in folders:
        if model_name not in aggregator[exp_type]:
            aggregator[exp_type][model_name] = {}
            # Each model will hold a dict of each metric
            for metric in score_types:
                aggregator[exp_type][model_name][metric] = {lang: [] for lang in LANG_LIST}

        # For each CSV file:
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
            if any(excl in file for excl in EXCLUDE_FILES):
                continue

            csv_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue

            # For each metric, gather data from the matching columns (e.g., "en_ROUGE-L")
            for metric in score_types:
                for lang in LANG_LIST:
                    col = f"{lang}_{metric}"
                    if col not in df.columns:
                        continue
                    else:
                        if 'en' not in col:
                            print(file)
                    aggregator[exp_type][model_name][metric][lang].extend(df[col].tolist())

    return aggregator

# ============== 3) Build a Metric-vs-Language DataFrame ==============
LANG_GROUPS = {
    "English": {"en"},
    "Translated": {"es", "vi", "tr"},
    "Cross-lingual": {"st", "tn", "ty", "mai", "mg", "yo"}
}

def build_metric_group_dataframe(aggregator_subdict, score_types):
    """
    Given aggregator_subdict[model][metric][lang] => list of values,
    1) Compute the average across all models for each (metric, group).
    2) Return a DataFrame with rows = metrics, columns = language groups.
    """
    # Reverse mapping: lang -> group
    lang_to_group = {}
    for group, langs in LANG_GROUPS.items():
        for lang in langs:
            lang_to_group[lang] = group

    # Initialize group-level storage
    final_scores = {metric: {group: [] for group in LANG_GROUPS} for metric in score_types}

    for model_name, metrics_dict in aggregator_subdict.items():
        for metric in score_types:
            for lang, group in lang_to_group.items():
                if lang in metrics_dict[metric]:
                    final_scores[metric][group].extend(metrics_dict[metric][lang])

    # Compute mean for each metric-group
    for metric in score_types:
        for group in LANG_GROUPS:
            vals = final_scores[metric][group]
            final_scores[metric][group] = np.nanmean(vals) if vals else np.nan

    # Build DataFrame
    df_rows = []
    for metric in score_types:
        row_dict = {"Metric": metric}
        for group in LANG_GROUPS:
            row_dict[group] = final_scores[metric][group]
        df_rows.append(row_dict)

    df = pd.DataFrame(df_rows)
    df.set_index("Metric", inplace=True)

    # Scale if needed
    for metric in SCALE_TO_100:
        if metric in df.index:
            df.loc[metric] = df.loc[metric] * 100.0

    return df


# ============== 4) Plot Heatmap: Metric (Y axis) vs Language (X axis) ==============
def plot_metric_vs_lang_heatmap(df, experiment_type):
    """
    Plots a single heatmap where:
      - Rows = metrics
      - Columns = languages
      - Values in [0..100] for all metrics
    """
    # Figure size can adjust for number of metrics
    plt.figure(figsize=(6, 4 + 0.3*len(df)))

    ax = sns.heatmap(
        df,
        annot=True, fmt=".1f",   # show 1 decimal place
        cmap=custom_cmap,
        vmin=0, vmax=100,
        cbar=True,
        annot_kws={"size": 18}
    )

    ax.set_title(f"Prefix Probe {experiment_type.capitalize()}")
    ax.set_xlabel("")
    ax.set_ylabel("")

    out_file = f"Prefix_probe_heatmap_{experiment_type}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved {experiment_type} heatmap to {out_file}")

def main():
    # 1) Gather aggregator data for multiple metrics
    aggregator = load_and_process_data_all_metrics(ALL_SCORE_TYPES)

    # 2) For each experiment type, build a Metric-vs-Language DataFrame & plot
    for exp_type in ["one-shot", "zero-shot"]:
        if exp_type not in aggregator:
            continue

        df_metric_lang = build_metric_group_dataframe(aggregator[exp_type], ALL_SCORE_TYPES)
        print(f"\n=== {exp_type.capitalize()} - Metric vs. Language Data ===\n")
        print(df_metric_lang)
        df_metric_lang.to_csv("metric.csv", index=False)
        # Create a single heatmap with row=metrics, col=languages
        plot_metric_vs_lang_heatmap(df_metric_lang, experiment_type=exp_type)

if __name__ == "__main__":
    main()