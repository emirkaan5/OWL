import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("evals/olmo_eval.csv")  # <-- Replace with your actual file path

# Model name mapping
model_name_map = {
    'gpt-4o-2024-11-20': 'gpt4o',
    'Llama-3.1-405b': 'llama405b',
    'OLMo-2-1124-13B-Instruct': 'olmo2',
    'EuroLLM-9B-Instruct': 'eurollm'
}

# Filter only those models
df = df[df['model'].isin(model_name_map.keys())].copy()
df['model'] = df['model'].map(model_name_map)

# Get all *_eval columns
language_eval_columns = [col for col in df.columns if col.endswith('_eval')]
language_names = [col.replace('_eval', '') for col in language_eval_columns]

# Build long-form accuracy records
records = []
for lang, col in zip(language_names, language_eval_columns):
    temp_df = df[['model', col]].copy()
    temp_df = temp_df[temp_df[col].notna()]
    temp_df = temp_df.rename(columns={col: 'correct'})
    temp_df['language'] = lang
    records.append(temp_df)

lang_model_df = pd.concat(records)
# Before creating the heatmap, convert the 'correct' column to numeric
lang_model_df['correct'] = pd.to_numeric(lang_model_df['correct'], errors='coerce')

# Then continue with your groupby and pivoting
lang_model_mean = lang_model_df.groupby(['language', 'model'])['correct'].mean().reset_index()
# Mean accuracy per language per model

# Pivot for heatmap
lang_heatmap = lang_model_mean.pivot(index='language', columns='model', values='correct')

# Sort by mean accuracy
lang_heatmap['avg'] = lang_heatmap.mean(axis=1)
lang_heatmap = lang_heatmap.sort_values(by='avg', ascending=False).drop(columns='avg')

# Plot heatmap with BuPu colormap and narrower figure
plt.figure(figsize=(5.5, len(lang_heatmap) * 0.4))  # smaller width
sns.heatmap(lang_heatmap, annot=True, cmap="BuPu", fmt=".2f", cbar_kws={'label': 'Mean Accuracy'})
plt.title("Direct Probe Accuracy on Dataset Occurance")
plt.xlabel("Model")
plt.ylabel("Language")
plt.tight_layout()
plt.show()
plt.savefig("evals/graphs/olmo_heatmap.png")