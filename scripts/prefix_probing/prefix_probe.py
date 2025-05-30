    import os
import pandas as pd
import re
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
from bs4 import BeautifulSoup


def extract_output(text):
    soup = BeautifulSoup(text, 'html.parser')
    passages = soup.find_all('continuation')

    if passages:
        return max([passage.get_text() for passage in passages])
    else:
        return text


def predict(passages, word_counts, llm, lang, prompt_setting="zero-shot"):

    SYSTEM_PROMPT = "You are a helpful assistant. You follow every single instruction carefully."
    
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
        
    prompt = """
       You are provided with an incomplete passage from a book. Your task is to continue the passage *in the SAME LANGUAGE* with {word_count}. Your response should ONLY be the continuation, directly quoted from the book. Do not repeat or rephrase the input.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <continuation>Passage continuation</continuation>
    """

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1000)

    prompts = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.format(
                word_count=word_count,
                demo_passage=demo_passage,
                passage=passage
            ).strip()},
            ] for passage, word_count in zip(passages, word_counts)
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    batch_results = []
    for output in outputs:
        extract = extract_output(output.outputs[0].text)
        extract = extract.replace('\n', ' ')

        batch_results.append(extract)

    return batch_results


def split_sentence_in_half(sentence):
    words = sentence.split()  
    midpoint = len(words) // 2  

    first_half = ' '.join(words[:midpoint])
    second_half = ' '.join(words[midpoint:])

    return first_half, second_half, midpoint


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


def prefixProbe(csv_file_name, book_title, llm, model_name, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)
        df_out = pd.DataFrame()
        
        languages = ["en", "vi", "es", "tr"]
        for lang in languages:
            try:
                print(f'///running {lang}///')
                df_out[[f"{lang}_first_half", f"{lang}_second_half", f"{lang}_word_count"]] = df[lang].apply(
                    lambda x: pd.Series(split_sentence_in_half(x)) if pd.notnull(x) else pd.Series([x, x, 0])
                )
                passages = df_out[f"{lang}_first_half"].tolist()
                word_counts = df_out[f"{lang}_word_count"].tolist()
                output = predict(passages, word_counts, llm, lang, prompt_setting)

                index_of_lang = df_out.columns.get_loc(f"{lang}_word_count")
                df_out.insert(index_of_lang + 1, f"{lang}_results_raw", pd.Series(output))
            except Exception as e:
                print(e)

        df_out.to_csv(f"{book_title}_prefix_probe_{model_name}.csv", index=False, encoding='utf-8')

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
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Name of the model to use")
    parser.add_argument("gpus", type=str, help="Nums of gpus to use")
    args = parser.parse_args()

    llm = LLM(model=args.model, tensor_parallel_size=int(args.gpus), max_model_len=2048)
    
    titles = get_folder_names('/scripts/Prompts')
    skip_list = []
    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            prefixProbe(csv_file_name=f"/scripts/Prompts/{title}/{title}_filtered.csv", book_title=title, llm=llm, model_name=args.model.split('/')[1], prompt_setting="zero-shot") # modify the prompt setting here