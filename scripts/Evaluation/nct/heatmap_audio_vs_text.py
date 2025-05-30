# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re

# BASE_PATH = "scripts/Evaluation/nct/eval"
# MODALITIES = ["text", "audio"]
# ALLOWED_MODELS = {
#     "GPT-4o": ["GPT-4o"],
#     "Qwen-2.5-Omni-7b": ["Qwen-2.5-Omni-7b", "Qwen-2.5-Omni-7B"]
# }
# # Updated regex: allow optional _audio before _eval
# CSV_REGEX = re.compile(r"name_cloze_(.+?)_one-shot(?:_audio)?_eval\.csv", re.IGNORECASE)

# def find_eval_files():
#     result = []
#     for modality in MODALITIES:
#         folder = os.path.join(BASE_PATH, modality)
#         if not os.path.exists(folder):
#             continue
#         for fname in os.listdir(folder):
#             m = CSV_REGEX.match(fname)
#             if not m:
#                 continue
#             model_raw = m.group(1)
#             for canonical, variants in ALLOWED_MODELS.items():
#                 if any(v.lower() in model_raw.lower() for v in variants):
#                     result.append({
#                         "modality": modality,
#                         "path": os.path.join(folder, fname),
#                         "filename": fname,
#                         "model": canonical
#                     })
#     return result

# def compute_accuracy(csv_path):
#     df = pd.read_csv(csv_path)
#     col = "en_correct"
#     if col not in df:
#         return None
#     ser = df[col].astype(str).str.lower()
#     acc = (ser == "correct").mean() * 100
#     return acc

# # --- Main workflow ---
# eval_files = find_eval_files()

# print("Discovered files:")
# for fileinfo in eval_files:
#     print(f"{fileinfo['modality']}: {fileinfo['filename']}")

# data = []
# for fileinfo in eval_files:
#     acc = compute_accuracy(fileinfo['path'])
#     if acc is not None:
#         data.append({
#             "Model": fileinfo['model'],
#             "Modality": fileinfo['modality'],
#             "Accuracy": acc
#         })

# df = pd.DataFrame(data)
# print(df)  # Debug: see if audio/text both present

# pivot = df.pivot(index="Model", columns="Modality", values="Accuracy").reindex(
#     ["GPT-4o", "Qwen-2.5-Omni-7b"]
# )

# # Heatmap
# plt.figure(figsize=(5, 2.5))
# sns.heatmap(
#     pivot,
#     annot=True,
#     fmt=".1f",
#     cmap="YlGnBu",
#     cbar_kws={'label': 'Accuracy (%)'},
#     linewidths=0.5,
#     vmin=0, vmax=100
# )
# plt.title("Accuracy of en_correct (Name Cloze)\n(GPT-4o and Qwen-2.5-Omni-7b)")
# plt.ylabel("Model")
# plt.xlabel("Modality")
# plt.tight_layout()
# plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------- Config ---------
BASE_PATH = "scripts/Evaluation/nct/eval"
MODALITIES = ["text", "audio"]
MODELS = ["GPT-4o", "Qwen-2.5-Omni-7b"]
COL = "en_correct"

COLORS = {
    "overlap": "#551A8B",
    "text_only": "#8EBAE5",
    "audio_only": "#BEB0D9",
    "neither": "#e0e0e0"
}

def find_model_file(modality, model):
    folder = os.path.join(BASE_PATH, modality)
    # Match: direct_probe_{MODEL}_one-shot(_audio)?_eval.csv
    for fname in os.listdir(folder):
        if model.lower().replace('-', '').replace('.', '').replace(' ', '') in fname.lower().replace('-', '').replace('.', '').replace(' ', ''):
            if fname.endswith("_eval.csv"):
                return os.path.join(folder, fname)
    return None

def get_outcome_counts(text_df, audio_df):
    text_flag = text_df[COL].astype(str).str.lower().eq("correct")
    audio_flag = audio_df[COL].astype(str).str.lower().eq("correct")
    overlap = (text_flag & audio_flag).sum()
    text_only = (text_flag & ~audio_flag).sum()
    audio_only = (~text_flag & audio_flag).sum()
    neither = (~text_flag & ~audio_flag).sum()
    total = len(text_flag)
    return overlap, text_only, audio_only, neither, total

def annotate_stacked_bars(ax, df, x, width=0.6, min_height=5):
    # min_height: don't put label inside if the segment is too thin (percent)
    for i, model in enumerate(df.index):
        cumulative = 0
        for part in ["overlap", "text_only", "audio_only", "neither"]:
            val = df.loc[model, part]
            if val < 0.1:  # skip zero segments
                cumulative += val
                continue
            y = cumulative + val / 2
            txt = f"{val:.1f}%"
            # If bar segment is too thin, put label just above
            if val < min_height:
                ax.text(
                    x[i], cumulative + val + 1.5,
                    txt,
                    ha="center", va="bottom", fontsize=11, fontweight="bold"
                )
            else:
                ax.text(
                    x[i], y,
                    txt,
                    ha="center", va="center", color="white" if part in ["overlap"] else "black",
                    fontsize=11, fontweight="bold"
                )
            cumulative += val


rows = []
for model in MODELS:
    text_file = find_model_file("text", model)
    audio_file = find_model_file("audio", model)
    if text_file and audio_file:
        text_df = pd.read_csv(text_file)
        audio_df = pd.read_csv(audio_file)
        overlap, text_only, audio_only, neither, total = get_outcome_counts(text_df, audio_df)
        rows.append({
            "model": model,
            "overlap": overlap,
            "text_only": text_only,
            "audio_only": audio_only,
            "neither": neither,
            "total": total
        })
    else:
        print(f"Missing file for {model}: {text_file}, {audio_file}")

df = pd.DataFrame(rows).set_index("model")
# Convert to percent
for col in ["overlap", "text_only", "audio_only"]:
    df[col] = df[col] / df["total"] * 100

# Plot
plt.figure(figsize=(8, 7))
x = np.arange(len(df))
width = 0.6

plt.bar(x, df["overlap"], width, color=COLORS["overlap"], label="Correct in Both")
plt.bar(x, df["text_only"], width, bottom=df["overlap"], color=COLORS["text_only"], label="Text only")
plt.bar(x, df["audio_only"], width, bottom=df["overlap"] + df["text_only"], color=COLORS["audio_only"], label="Audio only")
plt.bar(x, df["neither"], width, bottom=df["overlap"] + df["text_only"] + df["audio_only"], color=COLORS["neither"], label="Neither")
annotate_stacked_bars(plt.gca(), df, x)
plt.xticks(x, df.index, rotation=0, fontsize=13, fontweight="bold")
plt.ylabel("Percentage of Predictions (%)", fontsize=13)
plt.ylim(0, 100)
plt.title("Name Cloze (Text vs Audio)", fontsize=14, fontweight="bold")
plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=13)
plt.subplots_adjust(right=0.7)  
plt.savefig("NCT_audio_vs_text.png", dpi=500)
plt.show()
