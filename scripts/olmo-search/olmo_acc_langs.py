import pandas as pd
import os
import re
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# Set global font to bold
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 14  # Increase base font size
LANG_GROUPS = {
    "English": ["en_results_both_match"],
    "Translations": ["es_results_both_match", "vi_results_both_match", "tr_results_both_match"],
    "Crosslingual": "rest",  # rest will still be auto-detected
}

def compute_grouped_accuracies(filtered_data: dict) -> pd.DataFrame:
    records = []

    for model, books in filtered_data.items():
        for book, df in books.items():
            columns = df.columns

            all_eval_cols = [col for col in columns if col.endswith("results_both_match")]

            crosslingual_cols = [
                col for col in all_eval_cols
                if col not in LANG_GROUPS["English"] + LANG_GROUPS["Translations"]
            ]

            groups = {
                "English": LANG_GROUPS["English"],
                "Translations": LANG_GROUPS["Translations"],
                "Crosslingual": crosslingual_cols
            }

            for group_name, lang_cols in groups.items():
                if not lang_cols:
                    continue
                for group_name, lang_cols in groups.items():
                    if not lang_cols:
                        continue

                    # Only use columns that actually exist
                    existing_cols = [col for col in lang_cols if col in df.columns]
                    if not existing_cols:
                        continue

                    values = df[existing_cols].values.flatten()
                    valid = pd.Series(values).dropna()
                    if len(valid) == 0:
                        continue

                    acc = (valid == True).mean()

                    records.append({
                        "model": model,
                        "book": book,
                        "group": group_name,
                        "accuracy": acc
                    })

                valid = pd.Series(values).dropna()
                if len(valid) == 0:
                    continue
                acc = (valid == True).mean()

                records.append({
                    "model": model,
                    "book": book,
                    "group": group_name,
                    "accuracy": acc
                })

    return pd.DataFrame.from_records(records)

import pandas as pd

def compute_accuracies_from_master_csv(master_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(master_csv_path)

    lang_groups = {
        "English": ["en_eval"],
        "Translations": ["es_eval", "vi_eval", "tr_eval"],
    }

    # Detect all *_eval columns
    eval_cols = [col for col in df.columns if col.endswith("_eval")]
    crosslingual = [col for col in eval_cols if col not in lang_groups["English"] + lang_groups["Translations"]]
    lang_groups["Crosslingual"] = crosslingual

    records = []

    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        for group, cols in lang_groups.items():
            existing_cols = [col for col in cols if col in model_df.columns]
            if not existing_cols:
                continue

            values = model_df[existing_cols].values.flatten()
            valid = pd.Series(values).dropna()
            if len(valid) == 0:
                continue

            acc = (valid == True).mean()
            records.append({
                "model": model,
                "group": group,
                "accuracy": acc
            })

    return pd.DataFrame.from_records(records)

def compute_grouped_accuracies_all_rows(filtered_data: dict) -> pd.DataFrame:
    records = []

    for model, books in filtered_data.items():
        # Collect all rows across all books for each group
        group_values = {
            "English": [],
            "Translations": [],
            "Crosslingual": []
        }

        for book, df in books.items():
            columns = df.columns
            all_eval_cols = [col for col in columns if col.endswith("results_both_match")]

            crosslingual_cols = [
                col for col in all_eval_cols
                if col not in LANG_GROUPS["English"] + LANG_GROUPS["Translations"]
            ]

            groups = {
                "English": LANG_GROUPS["English"],
                "Translations": LANG_GROUPS["Translations"],
                "Crosslingual": crosslingual_cols
            }

            for group_name, lang_cols in groups.items():
                # Only include existing columns
                existing_cols = [col for col in lang_cols if col in df.columns]
                if not existing_cols:
                    continue
                values = df[existing_cols].values.flatten()
                group_values[group_name].extend(pd.Series(values).dropna().tolist())

        # After all books are processed, compute true average per group
        for group_name, values in group_values.items():
            if not values:
                continue
            acc = (pd.Series(values) == True).mean()
            records.append({
                "model": model,
                "group": group_name,
                "accuracy": acc
            })

    return pd.DataFrame.from_records(records)


def plot_grouped_accuracies(df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="group", y="accuracy")
    plt.title("Aggregated Accuracy by Language Group (All Models) (removed occurences)", fontweight='bold')
    plt.ylim(0, 1)
    plt.xlabel("Group", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.savefig("omgwtf.png")
    plt.show()
# --- 🔍 Extract book title from any filename ---
def filter_all_models(results_base_dir: str, skip_dict: dict[str, set[int]]) -> tuple[dict, dict]:
    """
    Traverse model directories and apply filtering using skip_dict.
    Returns two nested dicts:
    - filtered_data:   {model_name: {book_title: DataFrame (filtered)}}
    - unfiltered_data: {model_name: {book_title: DataFrame (original)}}
    """
    filtered_data = {}
    unfiltered_data = {}

    for model_name in os.listdir(results_base_dir):
        model_dir = os.path.join(results_base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        if "aggregate_heatmaps" in model_name:
            continue

        filtered_model = {}
        unfiltered_model = {}

        # Look for any 'evaluation' subfolders under all experiment settings
        for root, dirs, files in os.walk(model_dir):
            if not root.endswith("evaluation"):
                continue

            file_dict = {
                fname: os.path.join(root, fname)
                for fname in files
                if fname.endswith(".csv")
            }

            for fname, full_path in file_dict.items():
                book_key = extract_book_title(fname)
                df = pd.read_csv(full_path)

                skip_idxs = skip_dict.get(book_key, set())
                df_filtered = df.drop(index=skip_idxs, errors="ignore")

                filtered_model[book_key] = df_filtered
                unfiltered_model[book_key] = df

        filtered_data[model_name] = filtered_model
        unfiltered_data[model_name] = unfiltered_model

        print(f"✅ {model_name}: {len(filtered_model)} books processed.")

    return filtered_data, unfiltered_data

def extract_book_title(filename: str) -> str:
    """
    Extract the book title from a filename by removing known suffixes, model names, and variants.
    """
    stem = Path(filename).stem
    # Remove any known suffix like direct_probe_... or non_NE, etc.
    stem = re.sub(r'_direct_probe_.*', '', stem)
    stem = re.sub(r'_non_NE$', '', stem)
    stem = re.sub(r'_unmasked_passages$', '', stem)
    return stem.strip("_")


# --- 📌 Load master CSV and organize skip indices ---
def build_skip_dict(master_csv_path: str) -> dict[str, set[int]]:
    master_df = pd.read_csv(master_csv_path)
    skip_dict = {}
    for _, row in master_df.iterrows():
        key = extract_book_title(row["filename"])
        skip_dict.setdefault(key, set()).add(row["index"])
    return skip_dict


# --- 📁 Build file dictionary from directory ---
def build_file_dict(data_dir: str) -> dict[str, str]:
    return {
        fname: os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".csv")
    }


# --- 🧹 Filter data files using skip_dict ---
def filter_csv_files(file_dict: dict[str, str], skip_dict: dict[str, set[int]]) -> dict[str, pd.DataFrame]:
    filtered_data = {}
    for fname, full_path in file_dict.items():
        book_key = extract_book_title(fname)
        skip_idxs = skip_dict.get(book_key, set())

        if skip_idxs:
            df = pd.read_csv(full_path)
            df_filtered = df.drop(index=skip_idxs, errors="ignore")
            filtered_data[full_path] = df_filtered
            print(f"✅ Filtered: {fname} (skipped {len(skip_idxs)} rows)")
        else:
            print(f"➖ No rows to skip for: {fname}")
            df = pd.read_csv(full_path)
            filtered_data[full_path] = df
    return filtered_data


def plot_accuracy_comparison_bar(df_master_acc: pd.DataFrame, df_filtered_acc: pd.DataFrame):
    df_master_acc = df_master_acc.copy()
    df_filtered_acc = df_filtered_acc.copy()


    df_master_acc.loc[df_master_acc["group"].isin(["Translations", "Crosslingual"]), "accuracy"] = 0
    df_master_acc["source"] = "Occured in Dataset"
    df_filtered_acc["source"] = "Did not Occur in Dataset"

    combined = pd.concat([df_master_acc, df_filtered_acc], ignore_index=True)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=combined, x="group", y="accuracy", hue="source", palette="BuPu")

    # Add grid
    ax.yaxis.grid(True, linestyle=":", linewidth=1, alpha=0.6)
    ax.set_axisbelow(True)

    # Remove the box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add text annotations above each bar with improved styling
    # Add text annotations on top of each bar in black
# Add text annotations just above each bar in black
    # Add text annotations directly on top of each bar in black
    for p in ax.patches:
        height = p.get_height()
        if height < 0.01:
            continue

        ax.text(
            p.get_x() + p.get_width() / 2.,
            height-p.get_height()/2-0.015,  # top of the bar
            f'{height:.2f}',
            ha='center',
            va='bottom',  # attach text at bottom edge to sit directly above
            fontsize=12,
            fontweight='bold',
            color='black'  # use black for both light and dark bars
        )



    plt.title("Accuracy Comparison on DP non_ne: Occurances vs Non-Occurances Olmo 13b", fontweight='bold')
    plt.ylim(0, 1.1)  # Increased ylim to make room for labels
    plt.ylabel("Mean Accuracy", fontweight='bold')
    plt.xlabel(" ", fontweight='bold')
    
    # Format ticks
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    
    # Format legend
    legend = ax.legend(title="Source", fontsize=14, title_fontsize=14)
    plt.setp(legend.get_title(), fontweight='bold')
    for text in legend.get_texts():
        text.set_fontweight('bold')
        
    plt.tight_layout()
    plt.savefig("accuracy_comparison_barplot11.png", dpi=300, bbox_inches='tight')
    plt.show()

def compute_flat_accuracies_from_master_csv(master_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(master_csv_path)

    lang_groups = {
        "English": ["en_eval"],
        "Translations": ["es_eval", "vi_eval", "tr_eval"],
    }

    # Detect all *_eval columns
    eval_cols = [col for col in df.columns if col.endswith("_eval")]
    crosslingual = [col for col in eval_cols if col not in lang_groups["English"] + lang_groups["Translations"]]
    lang_groups["Crosslingual"] = crosslingual

    records = []

    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        for group, cols in lang_groups.items():
            existing_cols = [col for col in cols if col in model_df.columns]
            if not existing_cols:
                continue

            values = model_df[existing_cols].values.flatten()
            valid = pd.Series(values).dropna()

            if len(valid) == 0:
                continue

            acc = (valid == True).mean()
            records.append({
                "model": model,
                "group": group,
                "accuracy": acc
            })

    return pd.DataFrame.from_records(records)



# --- 🚀 Main execution ---
if __name__ == "__main__":
    master_csv_path = "/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/olmo_eval_non_ne_olmo13b.csv"
    data_dir = "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/EuroLLM-9B-Instruct/ne_zero_shot"
    results_dir = "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/OLMo-2-1124-13B-Instruct/non_ne_one_shot/"
    # skip_dict = build_skip_dict(master_csv_path)
    # file_dict = build_file_dict(data_dir)
    # filtered_data = filter_csv_files(file_dict, skip_dict)

    
    skip_dict = build_skip_dict(master_csv_path)
    all_filtered_data,unfiltered = filter_all_models(results_dir, skip_dict)
    # print(all_filtered_data.keys())
    # print(unfiltered.keys())
    # print(all_filtered_data["Llama-3.1-405b"]["1984_eval"].keys())
    df_master_acc = compute_flat_accuracies_from_master_csv(master_csv_path)
    print(df_master_acc.head())
    grouped_df = compute_grouped_accuracies_all_rows(all_filtered_data)

    plot_grouped_accuracies(grouped_df)

    plot_accuracy_comparison_bar(df_master_acc, grouped_df)
