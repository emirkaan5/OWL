import os
import pandas as pd
import re

def extract_continuation(text):
    pattern = r".*passage.*?:\s*(.*)"
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        return text.strip()

def trim_common_prefix_suffix(string1, string2):
    string2 = extract_continuation(string2)
    string2 = re.sub(r'^[^a-zA-Z]+', '', string2)

    def clean_text(text):
        return re.sub(r'\W+', '', text)  

    cleaned_string1 = clean_text(string1)
    cleaned_string2 = clean_text(string2)

    for i in range(len(cleaned_string1)):
        suffix = cleaned_string1[i:]
        if cleaned_string2.startswith(suffix):
            match_position = 0
            count = 0
            for char in string2:
                if re.match(r'\w', char):  
                    count += 1
                match_position += 1
                if count == len(suffix):
                    break
            return string2[match_position:].strip()

    return string2

def remove_extra_suffix(text, limit):
    if len(text) <= limit:
        return text
    elif text[limit] == ' ':
        return text[:limit]
    else:
        next_space = text.find(' ', limit)
        if next_space == -1:  
            return text
        else:
            return text[:next_space]

def process_csv_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                for lang in ['en', 'vi', 'tr', 'es']:
                    try:
                        index_of_lang = df.columns.get_loc(f"{lang}_word_count")
                        df.insert(index_of_lang + 1, f"{lang}_results", df.apply(
                            lambda row: remove_extra_suffix(trim_common_prefix_suffix(row[f"{lang}_first_half"], row[f"{lang}_results_raw"]), len(row[f"{lang}_second_half"])), 
                            axis=1
                        ))
                    except Exception as e:
                        print(e)
                    
                    df.to_csv(file_path, index=False)
                    print(f"Processed and updated {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")