import os
import pandas as pd
from bs4 import BeautifulSoup
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    name_tag = soup.find('name')
    if name_tag:
        return name_tag.decode_contents()

    return None

def predict(lang, passages, llm, mode="unshuffled", prompt_setting="zero-shot"):

    SYSTEM_PROMPT = "You are a helpful assistant. You follow instructions carefully."

    demonstrations = {
        "es": {
            "unshuffled": "Hemos de agregar que quemaba tan hondamente el pecho de [MASK], que quizá había mayor verdad en el rumor que lo que nuestra moderna incredulidad nos permite aceptar.",
            "shuffled": "lo Hemos quemaba de verdad nos moderna rumor hondamente que que el quizá tan en el mayor había que agregar pecho [MASK], que aceptar. de incredulidad permite nuestra"
        },
        "tr": {
            "unshuffled": "Ve [MASK]'ın göğsünü o kadar derinden yaktı ki, belki de modern şüphemizin kabul etmeye meyilli olmadığı söylentide daha fazla gerçeklik vardı.",
            "shuffled": "ki, yaktı göğsünü gerçeklik vardı. meyilli söylentide belki fazla [MASK]'ın derinden olmadığı Ve kadar şüphemizin de kabul modern etmeye daha o"
        },
        "vi": {
            "unshuffled": "Và chúng ta tất phải thuật lại rằng nó đã nung đốt thành dấu hằn vào ngực [MASK] sâu đến nỗi có lẽ trong lời đồn kia có nhiều phần sự thực hơn là đầu óc đa nghi của chúng ta trong thời hiện đại có thể sẵn sàng thừa nhận.",
            "shuffled": "ta phải thuật trong ta trong lẽ thể đại nỗi có nhận. nung đa hằn nghi đốt đồn lời vào dấu sâu Và hơn có sự hiện [MASK] của có phần thực kia ngực sẵn chúng tất thời nhiều sàng chúng đầu rằng đến là lại thừa đã óc nó thành"
        },
        "en": {
            "unshuffled": "And we must needs say, it seared [MASK]'s bosom so deeply, that perhaps there was more truth in the rumor than our modern incredulity may be inclined to admit.",
            "shuffled": "admit. say, to inclined that the be more must so than it may needs modern we in rumor was deeply, incredulity perhaps our seared bosom there [MASK]'s And truth"
        }
    }

    demo = demonstrations.get(lang)[mode]
    
    demo_passage = ""
    if prompt_setting != "zero-shot":
        demo_passage = f"""
        
        Here is an example:
        <passage>{demo}</passage>
        <output>Hester</output>
        
        """
        
    prompt = """
       You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
        <output>Name</output>
    """

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)

    prompts = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.format(
                    demo_passage=demo_passage,
                    passage=passage
                ).strip()},
            ] for passage in passages
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    batch_results = []
    for output in outputs:
        extract = extract_output(output.outputs[0].text)
        if not extract:
            extract = output.outputs[0].text
        extract = extract.replace('\n', ' ')
        batch_results.append(extract)

    return batch_results

def name_cloze(csv_file_name, book_title, llm, model_name, prompt_setting="zero-shot"):
    try:
        df = pd.read_json(csv_file_name)

        for language in df.columns:
            if language != 'Single_ent':
                print(f'Running {language}')
                passages = df[language].tolist()
                mode = "shuffled" if "shuffled" in language.lower() else "unshuffled"
                base_language = language.split('_')[0]
                output = predict(base_language, passages, llm, mode, prompt_setting)

                index_of_language = df.columns.get_loc(language)
                df.insert(index_of_language + 1, f"{language}_results", pd.Series(output))

        df.to_csv(f"out/{book_title}_name_cloze_{model_name}.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(f'Error: {e}')

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
    data_path = ""
    filename =  os.path.basename(data_path).replace(".json","")
    name_cloze(data_path,filename,llm,args.model.split('/')[1],"one-shot")
