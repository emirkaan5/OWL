import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Folder containing all evaluation CSVs
eval_folder = 'scripts/Evaluation/prefix_probe/eval/unmasked'

# Custom model order
custom_model_order = [
    "GPT-4o",
    "Llama-3.1-405B",
    "Llama-3.1-70B-Instruct",
    "Llama-3.3-70B-Instruct",
    "OLMo-2-1124-13B-Instruct",
    "OLMo-2-1124-7B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct-1M",
    "EuroLLM-9B-Instruct",
    "All"
]

# Collect scores
model_scores = {}

for csv_file in os.listdir(eval_folder):
    if csv_file.endswith('_eval.csv'):
        model_name = csv_file.replace("prefix_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "")
        df = pd.read_csv(os.path.join(eval_folder, csv_file))

        en_score = df["en_ChrF++"].mean()
        vi_score = df["vi_ChrF++"].mean()
        es_score = df["es_ChrF++"].mean()
        tr_score = df["tr_ChrF++"].mean()

        model_scores[model_name] = {
            "English": en_score,
            "Translations": (vi_score + es_score + tr_score) / 3
        }

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)

# Convert to DataFrame
heatmap_df = pd.DataFrame.from_dict(model_scores, orient='index')

# Add "All" row as mean of all other models
heatmap_df.loc["All"] = heatmap_df.mean(numeric_only=True)

# Reindex to match custom order, even if some models are missing
heatmap_df = heatmap_df.reindex(custom_model_order)

# Plot heatmap
plt.figure(figsize=(5, 7.5))
sns.heatmap(heatmap_df, annot=True, cmap=custom_cmap, fmt=".1f", linewidths=0.5, vmax=100,
            annot_kws={"weight": "bold", "size": 20})
plt.title("Prefix Probe")
plt.xlabel("Language Group")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig("fig_PP.png", dpi=500, bbox_inches='tight')
plt.show()