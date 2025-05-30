import requests
import pandas as pd
import time
import os
# Azure Translator API credentials
endpoint = "https://api.cognitive.microsofttranslator.com/"
subscription_key = "" #TODO: Enter Key Here
region = "westus" #TODO: Enter Region Here

# Test languages
languages = {
    "st": "Sesotho",
    "yo": "Yorùbá",
    "tn": "Setswana (Tswana)",
    "ty": "Reo Tahiti (Tahitian)",
    "mai": "Maithili",
    "mg": "Malagasy",
    "dv": "Divehi (Dhivehi)"
}

#translate a batch of text with exponential backoff for rate limit handling.
def translate_batch_with_backoff(texts, to_lang, max_retries=5, initial_delay=5):
    url = f"{endpoint}translate?api-version=3.0&to={to_lang}"
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-Type': 'application/json'
    }
    body = [{"text": text} for text in texts]
    delay = initial_delay #initial delay in seconds

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 429:  # Too many requests
            retry_after = int(response.headers.get("Retry-After", delay))
            print(f"Rate limit hit. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            delay *= 2  # exponential backoff looks ok!
        else:
            response.raise_for_status()
            return [item['translations'][0]['text'] for item in response.json()]

    raise Exception("Max retries exceeded for translation request.")

# estimate cost by char 
def estimate_cost(df, languages, cost_per_million=10):
    total_characters = df['en'].str.len().sum()
    total_translation_characters = total_characters * len(languages)
    estimated_cost = (total_translation_characters / 1_000_000) * cost_per_million
    return total_translation_characters, estimated_cost

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names
    
if __name__ == "__main__":
    base_directory = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/"
    titles = [folder for folder in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, folder))]

    for title in titles:
        input_csv = f"{base_directory}/{title}/{title}_non_NE.csv"

        print(f"----------------- Translating {title} -----------------")

        #also check if en col exists: bug resolved
        df = pd.read_csv(input_csv)
        if 'en' not in df.columns:
            print(f"Error: The input CSV for {title} does not contain a column named 'En'. Skipping...")
            continue

        # estimate cost for the file
        total_characters, estimated_cost = estimate_cost(df, languages)
        print(f"File: {input_csv}")
        print(f"Total characters: {total_characters}")
        print(f"Estimated cost: ${estimated_cost:.2f}")

        # translating then overwrite original file
        batch_size = 10
        for lang_code, lang_name in languages.items():
            print(f"Translating to {lang_name}...")
            translations = []
            for i in range(0, len(df), batch_size):
                batch = df['en'][i:i + batch_size].tolist()
                translations.extend(translate_batch_with_backoff(batch, lang_code))
                time.sleep(5)  # delay between batches
            df[lang_name] = translations
            print(f"Completed translations for {lang_name}. Waiting before next language...")
            time.sleep(10) #delay in between languages

        df.to_csv(input_csv, index=False)
        print(f"Translation complete for {title}. Updated {input_csv}.")
