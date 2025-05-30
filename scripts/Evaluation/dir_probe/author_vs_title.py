import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_PATH = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval/text"
VARIANT_DIR = {
    "Standard": "unmasked",
    "Masked Named Character": "masked",
    "No Named Character": "non_ne"
}

# Color scheme for title-author-both comparison
COLORS = {
    "both": "#551A8B",
    "title_only": "#8EBAE5",
    "author_only": "#BEB0D9",
    "neither": "#e0e0e0"
}

# Clean model name for plotting
def clean_model_name(name):
    return name.replace("masked_", "").replace("non_NE_", "").replace("-Instruct", "")

# Compute outcome counts
def compare_title_author_both(df):
    if not all(col in df.columns for col in ["en_results_title_match", "en_results_author_match", "en_results_both_match"]):
        return 0, 0, 0, 0
    title = df["en_results_title_match"].fillna(False)
    author = df["en_results_author_match"].fillna(False)
    both = df["en_results_both_match"].fillna(False)

    overlap = both.sum()
    title_only = (title & ~both).sum()
    author_only = (author & ~both).sum()
    total = len(df)
    neither = total - (overlap + title_only + author_only)
    return overlap, title_only, author_only, neither, total

# Aggregate model performance
def aggregate_title_author(folder_path):
    rows = []
    for f in os.listdir(folder_path):
        if f.endswith("_eval.csv") and "quantized" not in f:
            df = pd.read_csv(os.path.join(folder_path, f))
            overlap, title_only, author_only, neither, total = compare_title_author_both(df)
            model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "")
            display_name = clean_model_name(model)
            rows.append({
                "model": display_name,
                "both": overlap,
                "title_only": title_only,
                "author_only": author_only,
                "neither": neither,
                "total": total
            })
    return pd.DataFrame(rows).set_index("model")

# Plot single bar per model with 100% = total predictions
def plot_title_author_breakdown(base_path=BASE_PATH):
    for variant, subdir in VARIANT_DIR.items():
        save_path = f"fig_title_author_breakdown_{variant.replace(' ', '_').lower()}.png"
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        folder = os.path.join(base_path, subdir)
        df = aggregate_title_author(folder)

        # Convert to percentages
        for col in ["both", "title_only", "author_only", "neither"]:
            df[col] = df[col] / df["total"] * 100

        df = df.sort_values("both", ascending=False)

        x = np.arange(len(df))
        width = 0.6
        print(df["title_only"])
        # Stack the bars
        ax.bar(x, df["both"], width, color=COLORS["both"], label="Title + Author")
        ax.bar(x, df["title_only"], width, bottom=df["both"], color=COLORS["title_only"], label="Title only")
        ax.bar(x, df["author_only"], width, bottom=df["both"] + df["title_only"], color=COLORS["author_only"], label="Author only")
        ax.bar(x, df["neither"], width, bottom=df["both"] + df["title_only"] + df["author_only"], color=COLORS["neither"], label="Neither")

        # Labels and styling
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage of All Predictions (%)", fontsize=15, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(f"Title vs. Author (100% = all guesses) - {variant}", fontsize=16, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Make y-tick labels bold
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
            tick.label1.set_fontsize(13)

        # Legend outside
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=13)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Show the plot
        plt.show()
        plt.close()
        print(f"✅ Title-author breakdown figure saved to {save_path}")

# Run the new version
plot_title_author_breakdown()
