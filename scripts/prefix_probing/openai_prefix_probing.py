import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import re


client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)


def extract_output(text):
    soup = BeautifulSoup(text, 'html.parser')
    passages = soup.find_all('continuation')

    if passages:
        return max([passage.get_text() for passage in passages])
    else:
        return None


def predict(passage, lang, word_count, prompt_setting="zero-shot"):

    demonstrations = {
        "es": {
            "first_half": "Hemos de agregar que quemaba tan hondamente el pecho de Hester, que quizá había",
            "second_half": "mayor verdad en el rumor que lo que nuestra moderna incredulidad nos permite aceptar."
        },
        "tr": {
            "first_half": "Ve Hester'ın göğsünü o kadar derinden yaktı ki, belki de",
            "second_half": "modern şüphemizin kabul etmeye meyilli olmadığı söylentide daha fazla gerçeklik vardı."
        },
        "vi": {
            "first_half": "Và chúng ta tất phải thuật lại rằng nó đã nung đốt thành dấu hằn vào ngực Hester sâu đến nỗi có lẽ trong lời",
            "second_half": "đồn kia có nhiều phần sự thực hơn là đầu óc đa nghi của chúng ta trong thời hiện đại có thể sẵn sàng thừa nhận."
        },
        "en": {
            "first_half": "And we must needs say, it seared Hester's bosom so deeply, that perhaps there",
            "second_half": "was more truth in the rumor than our modern incredulity may be inclined to admit."
        }
    }

    demo = demonstrations.get(lang)
    
    demo_passage = ""
    if prompt_setting != "zero-shot":
        demo_passage = f"""
        
        Here is an example:
        <passage>{demo['first_half']}</passage>
        <continuation>{demo['second_half']}</continuation>
        
        """
        
    prompt = f"""
       You are provided with an incomplete passage from a book. Your task is to continue the passage *in the SAME LANGUAGE* with {word_count}. Your response should ONLY be the continuation, directly quoted from the book. Do not repeat or rephrase the input.
        {demo_passage}

        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <continuation>Passage continuation</continuation>
    """

    completion = client.chat.completions.create(
        model="gpt-4o-2024-11-20",

        max_tokens=100,

        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    else:
        print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def split_sentence_in_half(sentence):
    words = sentence.split()  
    midpoint = len(words) // 2  

    first_half = ' '.join(words[:midpoint])
    second_half = ' '.join(words[midpoint:])

    return first_half, second_half, len(words) // 2


def trim_common_prefix_suffix(string1, string2):
    string2 = re.sub(r'^[^a-zA-Z0-9]+', '', string2)

    def clean_text(text):
        return re.sub(r'\W+', '', text)  

    cleaned_string1 = clean_text(string1).lower()
    cleaned_string2 = clean_text(string2).lower()

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


def prefixProbe(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)
        df_out = pd.DataFrame()

        languages = ["en", "vi", "es", "tr"]
        for lang in languages:
            if lang in df.columns:
                print(f'///running {lang}///')
                output = []
                for i in range(len(df)):
                    full_passage = df[lang].iloc[i]
                    first_half, second_half, word_count = split_sentence_in_half(full_passage)
                    try:
                        print(f"Running prompt for {lang}: {first_half}")
                        completion = predict(first_half, lang, word_count, prompt_setting)
                        trimmed_completion = remove_extra_suffix(trim_common_prefix_suffix(first_half, completion), len(second_half))
                        output.append([first_half, second_half, trimmed_completion])
                    except Exception as e:
                        output.append([first_half, str(e), False])
                
                output_df = pd.DataFrame(output, columns=[f'{lang}_first_half', f'{lang}_second_half', f'{lang}_Completion'])
                df_out = pd.concat([df_out, output_df], axis=1)

        df_out.to_csv(f"{book_title}_prefix_probe_gpt-4o-2024-11-20_{prompt_setting}.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(e)


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = ['Alice_in_Wonderland']

    for title in titles:
        print(f'----------------- running {title} -----------------')
        prefixProbe(csv_file_name=f"/Users/alishasrivastava/BEAM/scripts/Prompts/{title}/{title}_filtered.csv", book_title=title, prompt_setting="zero-shot") # modify the prompt setting here