import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def compute_accuracy_from_eval(df):
    """
    Compute accuracy for each *_both_match column.
    Returns a dict: {lang_variant: accuracy%}
    """
    acc_dict = {}
    for col in df.columns:
        if col.endswith("_both_match"):
            total = len(df)
            correct = df[col].sum()
            acc = 100 * correct / total if total > 0 else 0.0
            acc_dict[col.replace("_both_match", "")] = acc
    return acc_dict

def make_accuracy_heatmap(eval_folder, output_path="accuracy_heatmap.png"):
    all_files = [f for f in os.listdir(eval_folder) if f.endswith("_eval.csv") and "quantized" not in f and "Omni" not in f]
    heatmap_data = {}

    for file in all_files:
        file_path = os.path.join(eval_folder, file)
        df = pd.read_csv(file_path)

        # Extract model name from filename
        model = file.replace("direct_probe_", "").replace("_one-shot_eval.csv", "")
        
        acc_dict = compute_accuracy_from_eval(df)
        print(file)
        print(acc_dict)
        heatmap_data[model] = acc_dict

    # Create DataFrame with models as rows, langs as columns
    heatmap_df = pd.DataFrame.from_dict(heatmap_data, orient="index")

    # Optional: sort columns by preferred order
    preferred_order = [
        'en_results', 'es_results', 'tr_results', 'vi_results', 'st_results', 'yo_results',
        'tn_results', 'ty_results', 'mai_results', 'mg_results',
        'en_shuffled_results', 'es_shuffled_results', 'tr_shuffled_results', 'vi_shuffled_results',
        'st_shuffled_results', 'yo_shuffled_results', 'tn_shuffled_results',
        'ty_shuffled_results', 'mai_shuffled_results', 'mg_shuffled_results'
    ]
    existing = [col for col in preferred_order if col in heatmap_df.columns]
    remaining = [col for col in heatmap_df.columns if col not in existing]
    heatmap_df = heatmap_df[existing + remaining]

    # Sort models by average accuracy
    heatmap_df["average_accuracy"] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values("average_accuracy", ascending=False)
    heatmap_df = heatmap_df.drop(columns=["average_accuracy"])
    
    # Enforce custom language order (X-axis)
    preferred_order = [
        # 'en_results', 'es_results', 'tr_results', 'vi_results',
        'yo_results', 'st_results', 'tn_results', 'mg_results', 'ty_results', 'mai_results'
        # 'en_shuffled_results', 'es_shuffled_results', 'tr_shuffled_results', 'vi_shuffled_results',
        # 'st_shuffled_results', 'yo_shuffled_results', 'tn_shuffled_results',
        # 'ty_shuffled_results', 'mai_shuffled_results', 'mg_shuffled_results'
    ] # excluded lang will also be excluded from the heatmap 

    heatmap_df = heatmap_df[[col for col in preferred_order if col in heatmap_df.columns]]
    
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap=custom_cmap, vmin=0, vmax=100,
                cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size":20,  "weight": "bold"}, linewidths=0.5)
    plt.title("Direct Probe accuracy", fontsize=14)
    plt.xlabel("Language")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.show()
    plt.close()
    print(f"Saved heatmap to {output_path}")
    return heatmap_df

if __name__ == "__main__":
    make_accuracy_heatmap("/Users/minhle/BEAM/scripts/Evaluation/dir_probe/eval/text/unmasked", output_path="fig3_dp.png")
