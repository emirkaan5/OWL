from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os


client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)


def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    name_tag = soup.find('name')
    if name_tag:
        return name_tag.decode_contents()

    return None


def predict(lang, passage, mode="unshuffled", prompt_setting="zero-shot"):
    
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
    
    prompt = f"""
       You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
        <output>Name</output>
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

def name_cloze_task(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language != 'Single_ent':
                print(f'Running {language}')
                output = []
                mode = "shuffled" if "shuffled" in language.lower() else "unshuffled"
                for i in range(len(df)):
                    masked_passage = df[language].iloc[i]
                    base_language = language.split('_')[0]
                    content = predict(base_language, masked_passage, mode, prompt_setting)
                    print(f'{i}: {content}')
                    output.append(content)
                index_of_language = df.columns.get_loc(language)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", guess_results)
                
        df.to_csv(f"{book_title}_name_cloze__gpt-4o-2024-11-20_{prompt_setting}.csv", index=False, encoding='utf-8')
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
    titles = get_folder_names('/Prompts')
    
    for title in titles:
        print(f'----------------- running {title} -----------------')
        name_cloze_task(csv_file_name=f"/Users/alishasrivastava/BEAM/scripts/Prompts/{title}/{title}_filtered_masked.csv", book_title=title, prompt_setting="zero-shot") # modify the prompt setting here