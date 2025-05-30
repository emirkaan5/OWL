import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ───── CONFIG ───── #
eval_folder = 'scripts/Evaluation/nct/eval/text'
csv_files = [f for f in os.listdir(eval_folder) if f.endswith('_eval.csv') and "Omni" not in f]

LANG_GROUPS = {
    "English": ["en"],
    "Translations": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mai", "mg"]
}

custom_model_order = [
    "GPT-4o",
    "Llama-3.1-405B",
    "Llama-3.1-70B-Instruct",
    "Llama-3.3-70B-Instruct",
    "OLMo-2-1124-13B-Instruct",
    "Qwen2.5-7B-Instruct-1M",
    "OLMo-2-1124-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "EuroLLM-9B-Instruct",
    "all"
]

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)

# ───── ACCURACY LOGIC ───── #
def compute_group_accuracies(df):
    group_accuracies = {}
    for group_name, langs in LANG_GROUPS.items():
        total_correct, total_guesses = 0, 0
        for lang in langs:
            col = f"{lang}_correct"
            if col in df.columns:
                correct_series = df[col]
                total = correct_series.notna().sum()
                correct = (correct_series == "correct").sum()
                total_correct += correct
                total_guesses += total
        if total_guesses > 0:
            group_accuracies[group_name] = 100 * total_correct / total_guesses

    for group_name, langs in LANG_GROUPS.items():
        total_correct, total_guesses = 0, 0
        for lang in langs:
            col = f"{lang}_shuffled_correct"
            if col in df.columns:
                correct_series = df[col]
                total = correct_series.notna().sum()
                correct = (correct_series == "correct").sum()
                total_correct += correct
                total_guesses += total
        if total_guesses > 0:
            group_accuracies[f"{group_name}_shuffled"] = 100 * total_correct / total_guesses

    return group_accuracies

def compute_all_group_accuracies(csv_files):
    total_corrects = {}
    total_counts = {}

    for group_name in LANG_GROUPS:
        total_corrects[group_name] = 0
        total_counts[group_name] = 0
        total_corrects[f"{group_name}_shuffled"] = 0
        total_counts[f"{group_name}_shuffled"] = 0

    for file in csv_files:
        if "quantize" in file:
            continue
        df = pd.read_csv(os.path.join(eval_folder, file))
        for group_name, langs in LANG_GROUPS.items():
            for lang in langs:
                for is_shuffled in [False, True]:
                    suffix = "_shuffled" if is_shuffled else ""
                    col = f"{lang}{suffix}_correct"
                    if col in df.columns:
                        correct_series = df[col]
                        total = correct_series.notna().sum()
                        correct = (correct_series == "correct").sum()
                        key = f"{group_name}{suffix}"
                        total_corrects[key] += correct
                        total_counts[key] += total

    acc = {}
    for group in total_corrects:
        if total_counts[group] > 0:
            acc[group] = 100 * total_corrects[group] / total_counts[group]
    return acc

# ───── PLOTTING FUNCTIONS ───── #
def plot_accuracy_heatmap():
    accuracy_data = {}
    for file in csv_files:
        model_name = file.replace('name_cloze_', '').replace('_one-shot_eval.csv', '').replace("Meta-", '')
        if "quantize" in model_name:
            continue
        df = pd.read_csv(os.path.join(eval_folder, file))
        acc = compute_group_accuracies(df)
        accuracy_data[model_name] = acc

    accuracy_df = pd.DataFrame(accuracy_data).T

    # Add all row
    all_row = compute_all_group_accuracies(csv_files)
    accuracy_df.loc["all"] = pd.Series(all_row)

    # Reorder
    ordered_cols = ["English", "Translations", "Cross-lingual"]
    accuracy_df = accuracy_df[[col for col in ordered_cols if col in accuracy_df.columns]]

    ordered_rows = [m for m in custom_model_order if m in accuracy_df.index]
    remaining_rows = [m for m in accuracy_df.index if m not in ordered_rows]
    accuracy_df = accuracy_df.loc[ordered_rows + remaining_rows]

    # Plot
    plt.figure(figsize=(7, 7.5))
    sns.heatmap(
        accuracy_df,
        annot=True,
        fmt=".1f",
        vmax=100,
        cmap=custom_cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Accuracy'},
        annot_kws={"weight": "bold", "size": 20}
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Name cloze accuracy')
    plt.xlabel('Language Groups')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig("fig4_nct.png", dpi=500, bbox_inches='tight')
    plt.show()


def plot_group_bar_chart():
    acc = compute_all_group_accuracies(csv_files)
    lang_group_order = list(LANG_GROUPS.keys())  # x-axis labels
    bar_width = 0.35
    colors = {"Normal": "#bfd3e6", "Shuffled": "#8c6bb1"}

    normal_vals = [acc.get(group, 0) for group in lang_group_order]
    shuffled_vals = [acc.get(f"{group}_shuffled", 0) for group in lang_group_order]
    x = range(len(lang_group_order))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.bar(
        [i - bar_width / 2 for i in x], normal_vals,
        width=bar_width, label="Normal", color=colors["Normal"]
    )
    ax.bar(
        [i + bar_width / 2 for i in x], shuffled_vals,
        width=bar_width, label="Shuffled", color=colors["Shuffled"]
    )

    # Add percentage labels above bars
    for i, (norm, shuf) in enumerate(zip(normal_vals, shuffled_vals)):
        ax.text(i - bar_width / 2, norm + 1, f"{norm:.1f}%", ha='center', va='bottom',
                fontsize=24, fontweight='bold')
        ax.text(i + bar_width / 2, shuf + 1, f"{shuf:.1f}%", ha='center', va='bottom',
                fontsize=24, fontweight='bold')

    ax.set_xticks(list(x))
    ax.set_xticklabels(lang_group_order, rotation=15, fontsize=26, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=20, fontweight='bold')
    ax.set_ylim(0, 20)
    ax.set_title("Name Cloze", fontsize=20, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(fontsize=20)

    plt.tight_layout()
    output_path = "fig5_bar_nct_grouped.png"
    plt.savefig(output_path, dpi=500)
    plt.show()
    plt.close()
    print(f"Saved grouped bar chart to {output_path}")



# ───── FUNCTION CALLS ───── #
plot_accuracy_heatmap()
# plot_group_bar_chart()
