import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ======== CONFIGURATION ======== #
eval_folder = 'scripts/Evaluation/nct/eval/text'
csv_files = [f for f in os.listdir(eval_folder) if f.endswith('_eval.csv') and "Omni" not in f]

# Language codes to include and their order on the x-axis
LANGS = ["yo", "st", "tn", "mg", "ty", "mai"]

# Custom order for models (rows in heatmap)
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

# ======== ACCURACY FUNCTIONS ======== #
def compute_lang_accuracies(df):
    acc_dict = {}
    for lang in LANGS:
        correct_col = f"{lang}_correct"
        if correct_col in df.columns:
            correct_series = df[correct_col]
            total = correct_series.notna().sum()
            correct = (correct_series == "correct").sum()
            if total > 0:
                acc_dict[lang] = 100 * correct / total
    return acc_dict

def compute_all_row_langwise(csv_files, eval_folder):
    total_corrects = {lang: 0 for lang in LANGS}
    total_counts = {lang: 0 for lang in LANGS}

    for file in csv_files:
        model_name = file.replace('name_cloze_', '').replace('_one-shot_eval.csv', '').replace("Meta-", '')
        if "quantize" in model_name:
            continue
        df_model = pd.read_csv(os.path.join(eval_folder, file))
        for lang in LANGS:
            col = f"{lang}_correct"
            if col in df_model.columns:
                col_data = df_model[col]
                total = col_data.notna().sum()
                correct = (col_data == "correct").sum()
                total_corrects[lang] += correct
                total_counts[lang] += total

    return {
        lang: 100 * total_corrects[lang] / total_counts[lang]
        if total_counts[lang] > 0 else None
        for lang in LANGS
    }

# ======== BUILD ACCURACY DATAFRAME ======== #
accuracy_data = {}

for file in csv_files:
    model_name = file.replace('name_cloze_', '').replace('_one-shot_eval.csv', '').replace("Meta-", '')
    if "quantize" in model_name:
        continue
    df = pd.read_csv(os.path.join(eval_folder, file))
    lang_acc = compute_lang_accuracies(df)
    accuracy_data[model_name] = lang_acc

accuracy_df = pd.DataFrame(accuracy_data).T

# Add 'all' row
all_row = compute_all_row_langwise(csv_files, eval_folder)
accuracy_df.loc["all"] = pd.Series(all_row)

# ======== ORDERING ======== #
# Ensure all LANGS are columns and in desired order
ordered_cols = [lang for lang in LANGS if lang in accuracy_df.columns]
accuracy_df = accuracy_df[ordered_cols]

# Reorder rows by custom model order
available_order = [model for model in custom_model_order if model in accuracy_df.index]
remaining_models = [model for model in accuracy_df.index if model not in available_order]
accuracy_df = accuracy_df.loc[available_order + remaining_models]

# ======== HEATMAP PLOTTING ======== #
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)

plt.figure(figsize=(10, 8))
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
plt.title('Name Cloze Accuracy by Language')
plt.xlabel('Language')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig("fig3_nct.png", dpi=500, bbox_inches='tight')
plt.show()
