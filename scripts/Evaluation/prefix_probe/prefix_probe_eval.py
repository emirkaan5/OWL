import pandas as pd
import sacrebleu
import evaluate
from tqdm import tqdm

# Load metrics only once
bleurt = evaluate.load("bleurt")
rouge = evaluate.load("rouge")

def evaluate_csv(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df_out = pd.DataFrame(index=df.index)
    langs = ["en", "vi", "es", "tr"]

    for lang in langs:
        pred_col = f"{lang}_Completion"
        ref_col = f"{lang}_second_half"

        if pred_col not in df.columns or ref_col not in df.columns:
            continue

        valid_df = df[[pred_col, ref_col]].dropna()
        predictions = valid_df[pred_col].tolist()
        references = valid_df[ref_col].tolist()
        valid_indices = valid_df.index

        if not predictions:
            continue

        print(f"Computing metrics for {lang}...")
        
        # Per-sample metrics with tqdm
        bleu_scores = []
        chrf_scores = []
        for p, r in tqdm(zip(predictions, references), total=len(predictions), desc=f"{lang} BLEU/ChrF++"):
            bleu_scores.append(sacrebleu.sentence_bleu(p, [r]).score)
            chrf_scores.append(sacrebleu.sentence_chrf(p, [r]).score)

        # Batch compute vectorized metrics
        try:
            rouge_scores = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
            bleurt_scores = bleurt.compute(predictions=predictions, references=references)["scores"]
        except Exception as e:
            print(f"Error during batch BLEURT/ROUGE computation for {lang}: {e}")
            continue

        # Fill scores into output dataframe
        df_out.loc[valid_indices, f"{lang}_ROUGE-L"] = rouge_scores["rougeL"]
        df_out.loc[valid_indices, f"{lang}_BLEU"] = bleu_scores
        df_out.loc[valid_indices, f"{lang}_ChrF++"] = chrf_scores
        df_out.loc[valid_indices, f"{lang}_BLEURT"] = bleurt_scores

        # System-level row
        try:
            system_row = {
                f"{lang}_ROUGE-L": rouge_scores["rougeL"],
                f"{lang}_BLEU": sacrebleu.corpus_bleu(hypotheses=predictions, references=[references]).score,
                f"{lang}_ChrF++": sacrebleu.corpus_chrf(hypotheses=predictions, references=[references]).score,
                f"{lang}_BLEURT": sum(bleurt_scores) / len(bleurt_scores),
            }
            df_out.loc["System Scores", list(system_row.keys())] = list(system_row.values())
        except Exception as e:
            print(f"System-level score error for {lang}: {e}")

    df_out.to_csv(output_path, index=False)
    print(f"Saved evaluation to: {output_path}")

if __name__ == "__main__":
    input_csv = "EMNLP_results/prefix_probe/unmasked/text/GPT4o_unmasked_prefix_probe_one-shot.csv"  
    output_csv = "scripts/Evaluation/prefix_probe/eval/GPT4o_unmasked_prefix_probe_one-shot_eval.csv" 
    evaluate_csv(input_csv, output_csv)
