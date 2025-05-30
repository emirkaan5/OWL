# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # -------- config ------------------------------------------------------

# BASE_PATH = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval"
# VARIANT_DIR = {
#     "Standard": "unmasked",
#     "Masked Named Character": "masked",
#     "No Named Character": "non_ne"
# }
# CROSS_LINGUAL = ["st", "yo", "tn", "ty", "mai", "mg"]
# EN_COL = "en_results_both_match"

# COLORS = dict(
#     cl_only="#ffb347",   # orange
#     overlap="#8c6bb1",   # purple
#     en_only="#4c72b0"    # blue
# )

# # -------- core computation --------------------------------------------

# def overlap_counts(df):
#     cl_cols = [f"{l}_results_both_match" for l in CROSS_LINGUAL if f"{l}_results_both_match" in df.columns]
#     if not cl_cols or EN_COL not in df.columns:
#         return 0, 0, 0
#     en_ok = df[EN_COL]
#     cl_ok = df[cl_cols].any(axis=1)
#     A = (en_ok & cl_ok).sum()       # overlap
#     B = (~en_ok & cl_ok).sum()      # CL only
#     C = (en_ok & ~cl_ok).sum()      # EN only
#     return A, B, C

# def aggregate_variant(folder_path):
#     A_total = B_total = C_total = 0
#     rows = []
#     for f in os.listdir(folder_path):
#         if f.endswith("_eval.csv") and "quantized" not in f:
#             df = pd.read_csv(os.path.join(folder_path, f))
#             A, B, C = overlap_counts(df)
#             A_total += A; B_total += B; C_total += C
#             model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "").replace("masked_", "").replace("non_NE_", "")
#             rows.append({"model": model, "A": A, "B": B})
#     return A_total, B_total, C_total, pd.DataFrame(rows).set_index("model")

# # -------- plot 1: 2-row overlap bars ----------------------------------

# def plot_overlap_bars(base_path=BASE_PATH, save_path="fig_overlap_bars_vertical.png", show_values=True):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for ax, (variant, subdir) in zip(axes, VARIANT_DIR.items()):
#         folder = os.path.join(base_path, subdir)
#         A, B, C, _ = aggregate_variant(folder)
#         total = A + B + C
#         ax.set_ylim(0, total * 1.1)  # add headroom to avoid label cutoff

#         # Plot vertical stacked bars
#         cl_bar = ax.bar(0, A, color=COLORS["overlap"])
#         ax.bar(0, B, bottom=A, color=COLORS["cl_only"])

#         en_bar = ax.bar(1, A, color=COLORS["overlap"])
#         ax.bar(1, C, bottom=A, color=COLORS["en_only"])

#         if show_values:
#             # CL bar values
#             ax.text(0, A / 2, f"{A}", ha="center", va="center", fontsize=10, color="white")
#             ax.text(0, A + B / 2, f"{B}", ha="center", va="center", fontsize=10)

#             # EN bar values
#             ax.text(1, A / 2, f"{A}", ha="center", va="center", fontsize=10, color="white")
#             ax.text(1, A + C / 2, f"{C}", ha="center", va="center", fontsize=10, color="white")

#         ax.set_xticks([0, 1])
#         ax.set_xticklabels(["CL correct", "English correct"], fontweight='bold')
#         ax.set_title(variant, fontsize=14, fontweight='bold')
#         ax.grid(axis='y', linestyle='--', alpha=0.3)

#     # Add a proper legend outside plot
#     handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ["cl_only", "overlap", "en_only"]]
#     labels = ["Cross-lingual only", "Overlap", "English only"]
#     fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=11)

#     plt.suptitle("Overlap of Correct Answers: English vs. Cross-lingual", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # make room for title and legend
#     # plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print("✅ overlap‐bar figure saved to", save_path)

# # -------- plot 2: classic stacked bar (per model) ----------------------

# def plot_stacked_crosslingual_bars(base_path=BASE_PATH, save_path="fig_stacked_crosslingual.png"):
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
#     for ax, (variant, subdir) in zip(axes, VARIANT_DIR.items()):
#         folder = os.path.join(base_path, subdir)
#         _, _, _, df = aggregate_variant(folder)
#         x = range(len(df))
#         ax.bar(x, df["A"], color=COLORS["overlap"], label="also correct in EN")
#         ax.bar(x, df["B"], bottom=df["A"], color=COLORS["cl_only"], label="CL only")
#         ax.set_xticks(x)
#         ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10, fontweight='bold')
#         ax.set_title(variant, fontsize=14, fontweight='bold')
#         ax.set_ylabel("Cross-lingual correct (count)")
#         if ax == axes[0]:
#             ax.legend(fontsize=9, frameon=False)

#     plt.suptitle("Cross-lingual Correct Predictions (Per Model)", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     # plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print("✅ stacked-bar figure saved to", save_path)

# # -------- main ---------------------------------------------------------

# if __name__ == "__main__":
#     plot_overlap_bars()
#     plot_stacked_crosslingual_bars()


# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # -------- config ------------------------------------------------------

# BASE_PATH = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval"
# VARIANT_DIR = {
#     "Standard": "unmasked",
#     "Masked Named Character": "masked",
#     "No Named Character": "non_ne"
# }
# CROSS_LINGUAL = ["st", "yo", "tn", "ty", "mai", "mg"]
# EN_COL = "en_results_both_match"

# COLORS = dict(
#     cl_only="#ffb347",   # orange
#     overlap="#8c6bb1",   # purple
#     en_only="#4c72b0"    # blue
# )

# # -------- core computation --------------------------------------------

# def overlap_counts(df):
#     cl_cols = [f"{l}_results_both_match" for l in CROSS_LINGUAL if f"{l}_results_both_match" in df.columns]
#     if not cl_cols or EN_COL not in df.columns:
#         return 0, 0, 0
#     en_ok = df[EN_COL]
#     cl_ok = df[cl_cols].any(axis=1)
#     A = (en_ok & cl_ok).sum()       # overlap
#     B = (~en_ok & cl_ok).sum()      # CL only
#     C = (en_ok & ~cl_ok).sum()      # EN only
#     return A, B, C

# def aggregate_variant(folder_path):
#     A_total = B_total = C_total = 0
#     rows = []
#     for f in os.listdir(folder_path):
#         if f.endswith("_eval.csv") and "quantized" not in f:
#             df = pd.read_csv(os.path.join(folder_path, f))
#             A, B, C = overlap_counts(df)
#             A_total += A; B_total += B; C_total += C
#             model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "").replace("masked_", "").replace("non_NE_", "")
#             rows.append({"model": model, "A": A, "B": B})
#     return A_total, B_total, C_total, pd.DataFrame(rows).set_index("model")

# # -------- plot 1: 2-row overlap bars ----------------------------------

# def plot_overlap_bars(base_path=BASE_PATH, save_path="fig_overlap_bars_vertical.png", show_values=True):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for ax, (variant, subdir) in zip(axes, VARIANT_DIR.items()):
#         folder = os.path.join(base_path, subdir)
#         A, B, C, _ = aggregate_variant(folder)
#         total = A + B + C
#         ax.set_ylim(0, total * 1.1)  # add headroom to avoid label cutoff

#         # Plot vertical stacked bars
#         cl_bar = ax.bar(0, A, color=COLORS["overlap"])
#         ax.bar(0, B, bottom=A, color=COLORS["cl_only"])

#         en_bar = ax.bar(1, A, color=COLORS["overlap"])
#         ax.bar(1, C, bottom=A, color=COLORS["en_only"])

#         if show_values:
#             # CL bar values
#             ax.text(0, A / 2, f"{A}", ha="center", va="center", fontsize=10, color="white")
#             ax.text(0, A + B / 2, f"{B}", ha="center", va="center", fontsize=10)

#             # EN bar values
#             ax.text(1, A / 2, f"{A}", ha="center", va="center", fontsize=10, color="white")
#             ax.text(1, A + C / 2, f"{C}", ha="center", va="center", fontsize=10, color="white")

#         ax.set_xticks([0, 1])
#         ax.set_xticklabels(["CL correct", "English correct"], fontweight='bold')
#         ax.set_title(variant, fontsize=14, fontweight='bold')
#         ax.grid(axis='y', linestyle='--', alpha=0.3)

#     # Add a proper legend outside plot
#     handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ["cl_only", "overlap", "en_only"]]
#     labels = ["Cross-lingual only", "Overlap", "English only"]
#     fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=11)

#     plt.suptitle("Overlap of Correct Answers: English vs. Cross-lingual", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # make room for title and legend
#     # plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print("✅ overlap‐bar figure saved to", save_path)

# # -------- plot 2: classic stacked bar (per model) ----------------------

# def plot_stacked_crosslingual_bars(base_path=BASE_PATH, save_path="fig_stacked_crosslingual.png"):
#     fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
#     for ax, (variant, subdir) in zip(axes, VARIANT_DIR.items()):
#         folder = os.path.join(base_path, subdir)
#         _, _, _, df = aggregate_variant(folder)
#         x = range(len(df))
#         ax.bar(x, df["A"], color=COLORS["overlap"], label="also correct in EN")
#         ax.bar(x, df["B"], bottom=df["A"], color=COLORS["cl_only"], label="CL only")
#         ax.set_xticks(x)
#         ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10, fontweight='bold')
#         ax.set_title(variant, fontsize=14, fontweight='bold')
#         ax.set_ylabel("Cross-lingual correct (count)")
#         if ax == axes[0]:
#             ax.legend(fontsize=9, frameon=False)

#     plt.suptitle("Cross-lingual Correct Predictions (Per Model)", fontsize=16, fontweight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#     # plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print("✅ stacked-bar figure saved to", save_path)

# # -------- main ---------------------------------------------------------

# if __name__ == "__main__":
#     plot_overlap_bars()
#     plot_stacked_crosslingual_bars()


# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Configuration
# BASE_PATH = "/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval/text"
# VARIANT_DIR = {
#     "Standard": "unmasked",
#     "Masked Named Character": "masked",
#     "No Named Character": "non_ne"
# }
# CROSS_LINGUAL = ["st", "yo", "tn", "ty", "mai", "mg"]
# EN_COL = "en_results_both_match"
# COLORS = dict(cl_only="#80b1d3", overlap="#7b5ea6", en_only="#b35889")

# # Overlap count logic
# def overlap_counts(df):
#     cl_cols = [f"{l}_results_both_match" for l in CROSS_LINGUAL if f"{l}_results_both_match" in df.columns]
#     if not cl_cols or EN_COL not in df.columns:
#         return 0, 0, 0
#     en_ok = df[EN_COL]
#     cl_ok = df[cl_cols].any(axis=1)
#     A = (en_ok & cl_ok).sum()
#     B = (~en_ok & cl_ok).sum()
#     C = (en_ok & ~cl_ok).sum()
#     return A, B, C

# # Aggregate results per variant
# def aggregate_variant(folder_path):
#     rows = []
#     for f in os.listdir(folder_path):
#         if f.endswith("_eval.csv") and "quantized" not in f:
#             df = pd.read_csv(os.path.join(folder_path, f))
#             A, B, C = overlap_counts(df)
#             model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "").replace("masked_", "").replace("non_NE_", "")
#             rows.append({"model": model, "A": A, "B": B, "C": C})
#     return pd.DataFrame(rows).set_index("model")

# # Plot: one stacked bar per model (percent-based)
# def plot_stacked_percentage_bar(base_path=BASE_PATH, variant="Standard", save_path="fig_stacked_percent_singlebar.png"):
#     folder = os.path.join(base_path, VARIANT_DIR[variant])
#     df = aggregate_variant(folder)

#     df["total"] = df["A"] + df["B"] + df["C"]
#     df[["A", "B", "C"]] = df[["A", "B", "C"]].div(df["total"], axis=0) * 100
#     df["EN_total"] = df["A"] + df["C"]
#     df = df.sort_values("A", ascending=False)

#     fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.7), 6))
#     x = range(len(df))

#     ax.bar(x, df["A"], color=COLORS["overlap"], label="Overlap")
#     ax.bar(x, df["B"], bottom=df["A"], color=COLORS["cl_only"], label="CL only")
#     ax.bar(x, df["C"], bottom=df["A"] + df["B"], color=COLORS["en_only"], label="EN only")

#     ax.set_xticks(x)
#     ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=10, fontweight="bold")
#     ax.set_ylabel("Percentage of Correct Predictions (%)", fontsize=12)
#     ax.set_ylim(0, 100)
#     ax.set_title(f"Correct Predictions Breakdown by Model (Stacked %)\n{variant}", fontsize=14, fontweight='bold')

#     handles = [plt.Rectangle((0,0),1,1,color=COLORS[k]) for k in ["cl_only", "overlap", "en_only"]]
#     labels = ["CL only", "Overlap", "EN only"]
#     ax.legend(handles, labels, loc="upper right", frameon=False)

#     plt.tight_layout()
#     # plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print("✅ single-bar percent figure saved to", save_path)

# # Generate plots for all variants
# for variant in VARIANT_DIR:
#     plot_stacked_percentage_bar(variant=variant)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BASE_PATH = "scripts/Evaluation/dir_probe/eval/text"
VARIANT_DIR = {
    "Standard": "unmasked",
    "Masked Named Character": "masked",
    "No Named Character": "non_ne"
}
CROSS_LINGUAL = ["st", "yo", "tn", "ty", "mai", "mg"]
EN_COL = "en_results_both_match"

# Color scheme
COLORS = {
    "overlap": "#551A8B",
    "cl_only": "#BEB0D9",
    "en_only": "#8EBAE5",
    "neither": "#e0e0e0"
}

# Clean model name for plotting
def clean_model_name(name):
    return name.replace("masked_", "").replace("non_NE_", "").replace("-Instruct", "")

# Overlap analysis
def overlap_counts(df):
    cl_cols = [f"{l}_results_both_match" for l in CROSS_LINGUAL if f"{l}_results_both_match" in df.columns]
    if not cl_cols or EN_COL not in df.columns:
        return 0, 0, 0, 0
    cl_correct_mask = df[cl_cols].any(axis=1)
    en_correct_mask = df[EN_COL]
    overlap = (en_correct_mask & cl_correct_mask).sum()
    cl_only = (~en_correct_mask & cl_correct_mask).sum()
    en_only = (en_correct_mask & ~cl_correct_mask).sum()
    total = len(df)
    return overlap, cl_only, en_only, total

# Aggregate model performance
def aggregate_variant(folder_path):
    rows = []
    for f in os.listdir(folder_path):
        if f.endswith("_eval.csv") and "quantized" not in f:
            df = pd.read_csv(os.path.join(folder_path, f))
            overlap, cl_only, en_only, total = overlap_counts(df)
            model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "")
            display_name = clean_model_name(model)
            rows.append({
                "model": display_name,
                "overlap": overlap,
                "cl_only": cl_only,
                "en_only": en_only,
                "neither": total - (overlap + cl_only + en_only),
                "total": total
            })
    return pd.DataFrame(rows).set_index("model")

# New: plot single bar per model with 100% = total predictions
def plot_outcome_distribution(base_path=BASE_PATH):
    for variant, subdir in VARIANT_DIR.items():
        save_path = f"fig_outcome_distribution_{variant.replace(' ', '_').lower()}.png"
        plt.figure(figsize=(12, 7))
        ax = plt.gca()

        folder = os.path.join(base_path, subdir)
        df = aggregate_variant(folder)

        # Convert to percentages
        for col in ["overlap", "cl_only", "en_only", "neither"]:
            df[col] = df[col] / df["total"] * 100

        df = df.sort_values("overlap", ascending=False)

        x = np.arange(len(df))
        width = 0.6

        # Stack the bars
        ax.bar(x, df["overlap"], width, color=COLORS["overlap"], label="Overlap")
        ax.bar(x, df["en_only"], width, bottom=df["overlap"], color=COLORS["en_only"], label="EN only")
        ax.bar(x, df["cl_only"], width, bottom=df["overlap"] + df["en_only"], color=COLORS["cl_only"], label="CL only")
        ax.bar(x, df["neither"], width, bottom=df["overlap"] + df["en_only"] + df["cl_only"], color=COLORS["neither"], label="Neither")

        # Labels and styling
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage of All Predictions (%)", fontsize=15, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_title(f"Direct Probe accuracy (100% = all guesses) - {variant}", fontsize=16, fontweight="bold")
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
        print(f"✅ Outcome distribution figure saved to {save_path}")

# Run the new version
plot_outcome_distribution()