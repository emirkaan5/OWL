#!/usr/bin/env python3
import os
import pandas as pd
from kokoro import KPipeline
import soundfile as sf

# ——— CONFIG ———
# Root directory where your CSVs live (and any subdirectories)
CSV_ROOT_DIR = ""
# List the CSV column names you want to process
COLUMNS = [
    "en_masked"
    # add more column names as needed
]

# Kokoro TTS settings
LANG_CODE   = "a"        # Kokoro language code
VOICE       = "af_heart" # Kokoro voice name
SAMPLE_RATE = 24000       # Output sample rate
# —————————————————
pipeline = KPipeline(lang_code='a') 

def convert_csv_file(csv_path: str,fn : str):
    """
    For each non-null cell in specified COLUMNS, synthesize its text row-by-row,
    and save a WAV next to the CSV named <basename>_<column>_<row_index>.wav
    """
    df = pd.read_csv(csv_path)
    basename = os.path.splitext(csv_path)[0]
    print(basename)
    print(df.columns)
    for col in COLUMNS:
        if col not in df.columns:
            continue
        for row_idx, text in df[col].items():
            if pd.isna(text):
                continue
            t = str(text).strip()
            if not t:
                continue
            # Synthesize this single row text
            for _, _, audio in pipeline(t, voice=VOICE):
                wav_path = f"test/{fn}/{fn}_{col}_{row_idx}.wav"
                out_dir = os.path.dirname(wav_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                sf.write(wav_path, audio, SAMPLE_RATE)
                print(f"✔ Saved: {wav_path}")
                break  # only first chunk per row


def main():
    for dirpath, _, filenames in os.walk(CSV_ROOT_DIR):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                print(f"file being used {fn}")
                csv_file = os.path.join(dirpath, fn)
                convert_csv_file(csv_file,fn)

if __name__ == "__main__":
    main()
