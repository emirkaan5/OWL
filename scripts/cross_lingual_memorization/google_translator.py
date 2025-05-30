'''from google.cloud import translate_v2 as translate

def test_translation():
    client = translate.Client()
    text = "text here"
    target_language = "st"
    result = client.translate(text, target_language=target_language)
    print(f"Translated text: {result['translatedText']}")

test_translation()
'''
import pandas as pd
from google.cloud import translate_v2 as translate

def translate_text(passage, lang):
    client = translate.Client()
    answer = client.translate(passage, target_language=lang)
    final = answer['translatedText']
    return final

def process_csv(path):
    df = pd.read_csv(path)
    
    match_cols = [col for col in df.columns if col.endswith('_match')]
    print("columns i found with _match:", match_cols) #TODO: in output, check that all langs are found
    
    #for every match col where there is 'False', run gt
    for match_col in match_cols:
        lang_prefix = match_col.split('_')[0] #get lang
        gt_col = f"{lang_prefix}_gt" #make gt lang col
        
        """if gt_col in df.columns: #TODO: if you are re-running on a file, un hashtag this code!!! 
            df.drop(columns=[gt_col], inplace=True)"""
        
        df[gt_col] = None #if no translation needed, default will be None
        
        for idx, row in df.iterrows():
            match_value = row[match_col]
            if pd.notnull(match_value) and str(match_value).strip().upper() == "FALSE": #sometimes doesnt find false - stripping and skipping any None
                print(f"FALSE found at row {idx} in column {match_col}. Translation to {lang_prefix} started.")
                text_to_translate = row['en_masked']
                try:
                    translated_text = translate_text(text_to_translate, lang_prefix)
                    df.at[idx, gt_col] = translated_text
                    print(f"Translation done for row {idx} in column {match_col}.")
                except Exception as e:
                    print(f"Translation failed for row {idx}, text: {text_to_translate}. Error: {e}")
    
    df.to_csv(path, index=False)
    print(f"Overwrote csv with new cols and saved csv to: {path}")

if __name__ == '__main__':
    file_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/The_Picture_of_Dorian_Gray/The_Picture_of_Dorian_Gray.csv" #TODO: edit this path
    process_csv(file_path)
