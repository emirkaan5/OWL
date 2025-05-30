import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LANG_GROUPS = {
    "English": ["en"],
    "Translations": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mai", "mg"]
}

def compute_grouped_accuracy(df, group_name, lang_variants):
    """
    Compute total correct / total evaluated for a given language group and variant type.
    Example lang_variants = ["en_results", "es_results", ..., "mg_results"]
    """
    total_correct = 0
    total_total = 0

    for lang_col in lang_variants:
        match_col = f"{lang_col}_both_match"
        if match_col in df.columns:
            col_data = df[match_col]
            total_correct += col_data.sum()
            total_total += col_data.notna().sum()

    if total_total == 0:
        return 0.0
    return 100 * total_correct / total_total

def make_grouped_accuracy_heatmap(eval_folder, output_path="grouped_accuracy_heatmap.png"):
    all_files = [f for f in os.listdir(eval_folder) if f.endswith("_eval.csv") and "quantized" not in f and "Omni" not in f]
    heatmap_data = {}
    totals_by_group = {}  # To accumulate total correct/total guesses for 'All'

    for group_name in LANG_GROUPS:
        totals_by_group[group_name] = {'correct': 0, 'total': 0}
        totals_by_group[f"{group_name} (shuffled)"] = {'correct': 0, 'total': 0}

    for file in all_files:
        file_path = os.path.join(eval_folder, file)
        df = pd.read_csv(file_path)

        model = file.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "")
        model_acc = {}

        for group_name, langs in LANG_GROUPS.items():
            # Normal
            lang_variants = [f"{lang}_results" for lang in langs]
            total_correct = 0
            total_total = 0

            for lang_col in lang_variants:
                match_col = f"{lang_col}_both_match"
                if match_col in df.columns:
                    col_data = df[match_col]
                    total_correct += col_data.sum()
                    total_total += col_data.notna().sum()

            acc = 100 * total_correct / total_total if total_total > 0 else 0.0
            model_acc[group_name] = acc
            totals_by_group[group_name]['correct'] += total_correct
            totals_by_group[group_name]['total'] += total_total

            # Shuffled
            shuffled_variants = [f"{lang}_shuffled_results" for lang in langs]
            total_correct = 0
            total_total = 0

            for lang_col in shuffled_variants:
                match_col = f"{lang_col}_both_match"
                if match_col in df.columns:
                    col_data = df[match_col]
                    total_correct += col_data.sum()
                    total_total += col_data.notna().sum()

            acc_shuf = 100 * total_correct / total_total if total_total > 0 else 0.0
            model_acc[f"{group_name} (shuffled)"] = acc_shuf
            totals_by_group[f"{group_name} (shuffled)"]['correct'] += total_correct
            totals_by_group[f"{group_name} (shuffled)"]['total'] += total_total

        heatmap_data[model] = model_acc

    # Add 'All' row (aggregated totals)
    all_row = {}
    for group_col in totals_by_group:
        correct = totals_by_group[group_col]['correct']
        total = totals_by_group[group_col]['total']
        all_row[group_col] = 100 * correct / total if total > 0 else 0.0
    heatmap_data["All"] = all_row  # Add to heatmap data

    # Build heatmap dataframe
    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    # Separate 'All' row before sorting
    all_row_df = heatmap_df.loc[["All"]]
    other_models_df = heatmap_df.drop(index="All")

    # Sort models by average accuracy
    other_models_df["average_accuracy"] = other_models_df.mean(axis=1)
    other_models_df = other_models_df.sort_values("average_accuracy", ascending=False)
    other_models_df = other_models_df.drop(columns=["average_accuracy"])

    # Reattach 'All' row at the bottom
    heatmap_df = pd.concat([other_models_df, all_row_df])

    # Reorder columns
    group_order = [name for name in LANG_GROUPS]
    # Uncomment below if you want shuffled groups too
    # for name in LANG_GROUPS:
    #     group_order.append(f"{name} (shuffled)")

    heatmap_df = heatmap_df[[col for col in group_order if col in heatmap_df.columns]]

    # Plotting
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )

    plt.figure(figsize=(7, 7.5))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap=custom_cmap, vmin=0, vmax=100,
                cbar_kws={"label": "Accuracy (%)"},
                annot_kws={"size": 20, "weight": "bold"},
                linewidths=0.5)
    plt.title("Direct Probe Accuracy", fontsize=14)
    plt.xlabel("Language Group")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.show()
    plt.close()
    print(f"Saved grouped heatmap to {output_path}")
    return heatmap_df

if __name__ == "__main__":
    make_grouped_accuracy_heatmap(
        "scripts/Evaluation/dir_probe/eval/text/unmasked",
        output_path="fig4_dp.png"
    )
