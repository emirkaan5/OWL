import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LANG_GROUPS = {
    "Original\nEnglish": ["en"],
    "Official\nTranslations": ["es", "tr", "vi"],
    "Unseen\nTranslations": ["st", "yo", "tn", "ty", "mai", "mg"]
}

LANG_GROUPS_SHUFFLED = {
    "Original\nEnglish": ["en_shuffled"],
    "Official\nTranslations": ["es_shuffled", "tr_shuffled", "vi_shuffled"],
    "Unseen\nTranslations": ["st_shuffled", "yo_shuffled", "tn_shuffled", "ty_shuffled", "mai_shuffled", "mg_shuffled"]
}

def compute_grouped_accuracy(df, lang_variants):
    total_correct = 0
    total_total = 0
    for lang_col in lang_variants:
        match_col = f"{lang_col}_both_match"
        if match_col in df.columns:
            col_data = df[match_col]
            total_correct += col_data.sum()
            total_total += col_data.notna().sum()
    return total_correct, total_total

def process_variant(eval_folder):
    all_files = [f for f in os.listdir(eval_folder) if f.endswith("_eval.csv") and "quantized" not in f and "Omni" not in f]
    heatmap_data = {}
    totals_by_group = {g: {'correct': 0, 'total': 0} for g in LANG_GROUPS}
    
    for file in all_files:
        file_path = os.path.join(eval_folder, file)
        df = pd.read_csv(file_path)
        model = file.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "").replace("masked_", "").replace("non_NE_", "")
        model_acc = {}

        for group_name, langs in LANG_GROUPS.items():
            lang_variants = [f"{lang}_results" for lang in langs]
            correct, total = compute_grouped_accuracy(df, lang_variants)
            acc = 100 * correct / total if total > 0 else 0.0
            model_acc[group_name] = acc
            totals_by_group[group_name]['correct'] += correct
            totals_by_group[group_name]['total'] += total

        heatmap_data[model] = model_acc

    # Add 'All' row
    all_row = {}
    for group in LANG_GROUPS:
        correct = totals_by_group[group]['correct']
        total = totals_by_group[group]['total']
        all_row[group] = 100 * correct / total if total > 0 else 0.0
    heatmap_data["All"] = all_row

    # Build and sort
    df = pd.DataFrame.from_dict(heatmap_data, orient="index")
    all_row_df = df.loc[["All"]]
    df = df.drop(index="All")
    df["avg"] = df.mean(axis=1)
    df = df.sort_values("avg", ascending=False).drop(columns="avg")
    df = pd.concat([df, all_row_df])
    return df

def plot_variant_heatmap(base_path, variant_subfolder, title, output_path):
    variant_path = os.path.join(base_path, variant_subfolder)
    df = process_variant(variant_path)

    group_order = list(LANG_GROUPS.keys())
    cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )
    # Ensure column order
    df = df[[col for col in group_order if col in df.columns]]

    plt.figure(figsize=(7, max(5, 0.7 * len(df))))
    sns.heatmap(
        df, cmap=cmap, annot=True, fmt=".1f", vmin=0, vmax=100,
        linewidths=0.5, annot_kws={"size": 20, "weight": "bold"},
        cbar=True
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved heatmap to {output_path}")

def plot_three_heatmaps(dfs, variant_names, output_path):
    group_order = list(LANG_GROUPS.keys())
    cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

    for ax, df, name in zip(axes, dfs, variant_names):
        df = df[[col for col in group_order if col in df.columns]]
        sns.heatmap(
            df, ax=ax, cmap=cmap, annot=True, fmt=".1f",
            vmin=0, vmax=100, linewidths=0.5,
            cbar=(ax == axes[2]),  # only show colorbar on rightmost
            annot_kws={"size": 20, "weight": "bold"}
        )
        ax.set_title(name, fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved combined heatmap to {output_path}")
    
def plot_aggregated_variant_heatmap(base_path, output_path="aggregated_variant_heatmap.png"):
    variant_info = {
        "Standard": "unmasked",
        "Masked\nNamed Character": "masked",
        "No Named\nCharacter": "non_ne"
    }

    data = {}
    for variant_name, subfolder in variant_info.items():
        eval_folder = os.path.join(base_path, subfolder)
        all_files = [f for f in os.listdir(eval_folder) if f.endswith("_eval.csv") and "quantized" not in f and "Omni" not in f]

        totals_by_group = {group: {'correct': 0, 'total': 0} for group in LANG_GROUPS}

        for file in all_files:
            df = pd.read_csv(os.path.join(eval_folder, file))
            for group_name, langs in LANG_GROUPS.items():
                lang_variants = [f"{lang}_results" for lang in langs]
                for lang_col in lang_variants:
                    match_col = f"{lang_col}_both_match"
                    if match_col in df.columns:
                        col_data = df[match_col]
                        totals_by_group[group_name]['correct'] += col_data.sum()
                        totals_by_group[group_name]['total'] += col_data.notna().sum()

        # Compute final aggregated accuracy for this variant
        row = {}
        for group in LANG_GROUPS:
            correct = totals_by_group[group]['correct']
            total = totals_by_group[group]['total']
            acc = 100 * correct / total if total > 0 else 0.0
            row[group] = acc
        data[variant_name] = row

    # Convert to DataFrame
    heatmap_df = pd.DataFrame.from_dict(data, orient="index")
    heatmap_df = heatmap_df[[col for col in LANG_GROUPS.keys() if col in heatmap_df.columns]]  # column order

    # Plot
    cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap=cmap, vmin=0,
        linewidths=0.5, 
        annot_kws={"size": 26, "weight": "bold"},
        cbar_kws={"label": "Accuracy (%)"}
    )
    plt.title("Direct Probe Accuracy", fontsize=16, fontweight='bold')
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    ax.tick_params(axis='y', labelrotation=0)
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved aggregated heatmap to {output_path}")
    
def plot_grouped_bars_by_lang_group(base_path, output_path="barplot_by_lang_group.png"):
    variant_info = {
        "Original": "unmasked",
        "Masked": "masked",
        "No NE": "non_ne"
    }

    variant_order = list(variant_info.keys())
    lang_group_order = list(LANG_GROUPS.keys())

    # Data structure: {lang_group: {variant: {"normal": x, "shuffled": y}}}
    grouped_data = {group: {variant: {"normal": 0.0, "shuffled": 0.0} for variant in variant_order} for group in lang_group_order}

    for variant_name, subfolder in variant_info.items():
        eval_folder = os.path.join(base_path, subfolder)
        all_files = [f for f in os.listdir(eval_folder) if f.endswith("_eval.csv") and "quantized" not in f]

        totals = {group: {"normal": {"correct": 0, "total": 0}, "shuffled": {"correct": 0, "total": 0}} for group in lang_group_order}

        for file in all_files:
            df = pd.read_csv(os.path.join(eval_folder, file))
            for group_name, langs in LANG_GROUPS.items():
                normal_cols = [f"{lang}_results_both_match" for lang in langs]
                shuf_cols = [f"{lang}_shuffled_results_both_match" for lang in langs]
                for cols, mode in [(normal_cols, "normal"), (shuf_cols, "shuffled")]:
                    for col in cols:
                        if col in df.columns:
                            col_data = df[col]
                            totals[group_name][mode]["correct"] += col_data.sum()
                            totals[group_name][mode]["total"] += col_data.notna().sum()

        for group_name in lang_group_order:
            for mode in ["normal", "shuffled"]:
                correct = totals[group_name][mode]["correct"]
                total = totals[group_name][mode]["total"]
                acc = 100 * correct / total if total > 0 else 0.0
                grouped_data[group_name][variant_name][mode] = acc

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
    bar_width = 0.35
    colors = {"normal": "#bfd3e6", "shuffled": "#8c6bb1"}

    for ax, group_name in zip(axes, lang_group_order):
        normal_vals = [grouped_data[group_name][v]["normal"] for v in variant_order]
        shuffled_vals = [grouped_data[group_name][v]["shuffled"] for v in variant_order]
        x = range(len(variant_order))

        ax.bar([i - bar_width/2 for i in x], normal_vals, width=bar_width, label='Normal', color=colors["normal"])
        ax.bar([i + bar_width/2 for i in x], shuffled_vals, width=bar_width, label='Shuffled', color=colors["shuffled"])
        
        for i, (norm, shuf) in enumerate(zip(normal_vals, shuffled_vals)):
            ax.text(i - bar_width/2, norm + 1, f"{norm:.0f}%", ha='center', va='bottom', fontsize=18, fontweight='bold')
            ax.text(i + bar_width/2, shuf + 1, f"{shuf:.0f}%", ha='center', va='bottom', fontsize=18, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(variant_order, rotation=15, fontsize=20, fontweight='bold')
        ax.set_title(group_name, fontsize=24, fontweight='bold')
        ax.set_ylabel("Accuracy (%)" if ax == axes[0] else "", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 70)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    axes[2].legend(loc="upper right", fontsize=16)
    # plt.suptitle("Accuracy by Language Group and Direct Probe Variant", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved grouped bar chart to {output_path}")


if __name__ == "__main__":
    base_path = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval/text"

    df_masked   = process_variant(os.path.join(base_path, "masked"))
    df_unmasked = process_variant(os.path.join(base_path, "unmasked"))
    df_non_ne   = process_variant(os.path.join(base_path, "non_ne"))

    # plot_three_heatmaps(
    #     [df_unmasked, df_masked, df_non_ne],
    #     ["Unmasked", "Masked", "Non-NE"],
    #     output_path="fig6_dp_variations.png"
    # )
    
    # base_path = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval/text"
    # plot_variant_heatmap(base_path, "unmasked", "Unmasked", "fig4_dp.png")
    
    plot_aggregated_variant_heatmap(
        base_path,
        output_path="fig6_dp_aggregated.png"
    )
    
    # plot_grouped_bars_by_lang_group(
    #     base_path,
    #     output_path="fig6_barchart.png"
    # )