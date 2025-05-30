import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Configuration
BASE_PATH = "/Users/alishasrivastava/Desktop/graphs/eval/text"
VARIANT_DIR = {
    "Standard": "unmasked",
    "Masked Named Character": "masked",
    "No Named Character": "non_ne"
}
CROSS_LINGUAL = ["st", "yo", "tn", "ty", "mai", "mg"]
EN_COL = "en_results_both_match"

# Colors from the heatmap 
COLORS = {
    "overlap": "#551A8B",    # deep purple from heatmap (darkest)
    "cl_only": "#BEB0D9",    # pastel purple for CL only
    "en_only": "#8EBAE5"     # light blue from heatmap
}

def clean_model_name(name):
    """
    Clean model name by removing -Instruct and other prefixes
    using simple string replacement
    """
    # Apply all replacements in one go
    name = name.replace("masked_", "").replace("non_NE_", "").replace("-Instruct", "")
    return name

def compute_accuracy(df, column):
    """
    Compute accuracy for a column using the same approach as in heatmap_all.py
    """
    total = df[column].notna().sum()
    correct = df[column].sum()
    acc = 100 * correct / total if total > 0 else 0.0
    return acc, correct, total

def overlap_counts(df):
    """Calculate overlap, CL-only and EN-only correct counts"""
    cl_cols = [f"{l}_results_both_match" for l in CROSS_LINGUAL if f"{l}_results_both_match" in df.columns]
    if not cl_cols or EN_COL not in df.columns:
        return 0, 0, 0, 0, 0
    
    # For each example, check if any CL language is correct
    cl_correct_mask = df[cl_cols].any(axis=1)
    en_correct_mask = df[EN_COL]
    
    # Count overlaps and exclusives
    overlap = (en_correct_mask & cl_correct_mask).sum()
    cl_only = (~en_correct_mask & cl_correct_mask).sum()
    en_only = (en_correct_mask & ~cl_correct_mask).sum()
    
    # Compute total examples (used for percentage calculations)
    en_total = df[EN_COL].notna().sum()
    
    # Average of all CL languages' totals
    cl_totals = [df[col].notna().sum() for col in cl_cols]
    cl_total = sum(cl_totals) / len(cl_totals) if cl_totals else 0
    
    return overlap, cl_only, en_only, en_total, cl_total

def aggregate_variant(folder_path):
    """Aggregate results for all models in a variant folder"""
    rows = []
    for f in os.listdir(folder_path):
        if f.endswith("_eval.csv") and "quantized" not in f:
            df = pd.read_csv(os.path.join(folder_path, f))
            overlap, cl_only, en_only, en_total, cl_total = overlap_counts(df)
            
            # Get individual language accuracies using the original method
            cl_accs = {}
            for lang in CROSS_LINGUAL:
                col = f"{lang}_results_both_match"
                if col in df.columns:
                    acc, _, _ = compute_accuracy(df, col)
                    cl_accs[lang] = acc
            
            # Calculate average CL accuracy using the correct method
            cl_avg_acc = sum(cl_accs.values()) / len(cl_accs) if cl_accs else 0
            
            # Calculate EN accuracy using the correct method
            en_acc, _, _ = compute_accuracy(df, EN_COL)
            
            # Extract and clean model name exactly as in the original code
            model = f.replace("direct_probe_", "").replace("_one-shot_eval.csv", "").replace("Meta-", "").replace("masked_", "").replace("non_NE_", "")
            
            # Clean model name using simple string replacement for -Instruct
            display_name = clean_model_name(model)
            
            rows.append({
                "model": display_name,  # Use cleaned model name
                "overlap": overlap, 
                "cl_only": cl_only, 
                "en_only": en_only,
                "en_total": en_total,
                "cl_total": cl_total,
                "en_acc": en_acc,
                "cl_avg_acc": cl_avg_acc
            })
    return pd.DataFrame(rows).set_index("model")

def plot_stacked_comparison(base_path=BASE_PATH):
    """Plot stacked bars with overlap shown in common color, one plot per variant"""
    
    for variant, subdir in VARIANT_DIR.items():
        save_path = f"fig_stacked_comparison_{variant.replace(' ', '_').lower()}.png"
        
        # Create a figure for this variant
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        folder = os.path.join(base_path, subdir)
        df = aggregate_variant(folder)
        
        # Calculate totals
        df["cl_total_correct"] = df["overlap"] + df["cl_only"]
        df["en_total_correct"] = df["overlap"] + df["en_only"]
        
        # Sort by English accuracy
        df = df.sort_values("en_acc", ascending=False)
        
        # Set up plotting
        x = np.arange(len(df))
        width = 0.4
        
        # Plot EN bars (now on the left)
        ax.bar(x - width/2, df["overlap"], width, color=COLORS["overlap"], label="Overlap")
        ax.bar(x - width/2, df["en_only"], width, bottom=df["overlap"], color=COLORS["en_only"], label="EN only")
        
        # Plot CL bars (now on the right)
        ax.bar(x + width/2, df["overlap"], width, color=COLORS["overlap"])
        ax.bar(x + width/2, df["cl_only"], width, bottom=df["overlap"], color=COLORS["cl_only"], label="CL only")
        
        # Add percentage labels at the top of each bar (using the heatmap_all.py calculation method)
        for j, model in enumerate(df.index):
            # EN accuracy label (matching heatmap_all.py calculation) - now left
            # Shift label position slightly to the right (+0.05)
            ax.text(j - width/2 + 0.05, df.loc[model, "en_total_correct"] + 5, 
                    f'{df.loc[model, "en_acc"]:.1f}%', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            
            # CL average accuracy label (matching heatmap_all.py calculation) - now right
            # Shift label position slightly to the right (+0.05)
            ax.text(j + width/2 + 0.05, df.loc[model, "cl_total_correct"] + 5, 
                    f'{df.loc[model, "cl_avg_acc"]:.1f}%', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=15, fontweight='bold')
        ax.set_ylabel("Number of correct predictions", fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Make y tick labels bold
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
            tick.label1.set_fontsize(15)
        
        # Add thin black bar edges
        for bar in ax.patches:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
            
        # Remove the box and keep only x and y axes as light gray lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
            
        # Add legend
        ax.legend(fontsize=13, frameon=False, loc='upper right')
        
        plt.title(f"Cross-lingual vs. English Performance Comparison - {variant}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure without displaying it (to avoid interruption)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"✅ Stacked comparison figure saved to {save_path}")

def plot_percentage_stacked_comparison(base_path=BASE_PATH):
    """Plot stacked bars as percentages with overlap shown in common color, one plot per variant"""
    
    for variant, subdir in VARIANT_DIR.items():
        save_path = f"fig_percentage_stacked_comparison_{variant.replace(' ', '_').lower()}.png"
        
        # Create a figure for this variant
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        folder = os.path.join(base_path, subdir)
        df = aggregate_variant(folder)
        
        # Use same calculation as heatmap_all.py for percentages
        # Sort by English accuracy
        df = df.sort_values("en_acc", ascending=False)
        
        # Calculate percentage values for the bars
        df["overlap_pct"] = df["overlap"] / df["en_total"] * 100  # Normalized to EN total
        df["en_only_pct"] = df["en_only"] / df["en_total"] * 100  # Normalized to EN total
        df["cl_overlap_normalized"] = df["overlap"] / df["cl_total"] * 100  # Normalized to CL total
        df["cl_only_pct"] = df["cl_only"] / df["cl_total"] * 100  # Normalized to CL total
        
        # Set up plotting
        x = np.arange(len(df))
        width = 0.4
        
        # Plot EN bars (using normalized percentages) - now on the left
        ax.bar(x - width/2, df["overlap_pct"], width, color=COLORS["overlap"], label="Overlap")
        ax.bar(x - width/2, df["en_only_pct"], width, bottom=df["overlap_pct"], color=COLORS["en_only"], label="EN only")
        
        # Plot CL bars (using normalized percentages) - now on the right
        ax.bar(x + width/2, df["cl_overlap_normalized"], width, color=COLORS["overlap"])
        ax.bar(x + width/2, df["cl_only_pct"], width, bottom=df["cl_overlap_normalized"], color=COLORS["cl_only"], label="CL only")
        
        # Add percentage labels at the top of each bar (using the exact heatmap_all.py calculation)
        for j, model in enumerate(df.index):
            # EN accuracy label (from heatmap_all.py calculation) - now left
            # Shift label position slightly to the right (+0.05)
            ax.text(j - width/2 + 0.05, df.loc[model, "overlap_pct"] + df.loc[model, "en_only_pct"] + 1, 
                    f'{df.loc[model, "en_acc"]:.1f}%', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            
            # CL average accuracy label (from heatmap_all.py calculation) - now right
            # Shift label position slightly to the right (+0.05)
            ax.text(j + width/2 + 0.05, df.loc[model, "cl_overlap_normalized"] + df.loc[model, "cl_only_pct"] + 1, 
                    f'{df.loc[model, "cl_avg_acc"]:.1f}%', 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=15, fontweight='bold')
        ax.set_ylabel("Percentage of correct predictions (%)", fontsize=16, fontweight='bold')
        ax.set_ylim(0, 110)  # Increase y-limit to make room for the labels
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Make y tick labels bold
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
            tick.label1.set_fontsize(15)
        
        # Add thin black bar edges
        for bar in ax.patches:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.5)
            
        # Remove the box and keep only x and y axes as light gray lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')
            
        # Add legend
        ax.legend(fontsize=13, frameon=False, loc='upper right')
        
        plt.title(f"Cross-lingual vs. English Performance Comparison (%) - {variant}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure without displaying it (to avoid interruption)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"✅ Percentage stacked comparison figure saved to {save_path}")

if __name__ == "__main__":
    plot_stacked_comparison()
    plot_percentage_stacked_comparison() 