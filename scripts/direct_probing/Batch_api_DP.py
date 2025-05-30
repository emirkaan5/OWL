import os
import glob
import json
import time
import openai
import pandas as pd

# Define which CSV columns to process.
ALLOWED_COLUMNS = [
    "en", "en_shuffled"
    ]

def construct_prompt(lang, passage, mode, prompt_setting="zero-shot"):
    try:
        demonstrations = {
            "es": {
                "unshuffled": "Hemos de agregar que quemaba tan hondamente el pecho de Hester, que quizá había mayor verdad en el rumor que lo que nuestra moderna incredulidad nos permite aceptar.",
                "shuffled": "lo Hemos quemaba de verdad nos moderna rumor hondamente que que el quizá tan en el mayor había que agregar pecho Hester, que aceptar. de incredulidad permite nuestra"
            },
            "tr": {
                "unshuffled": "Ve Hester'ın göğsünü o kadar derinden yaktı ki, belki de modern şüphemizin kabul etmeye meyilli olmadığı söylentide daha fazla gerçeklik vardı.",
                "shuffled": "ki, yaktı göğsünü gerçeklik vardı. meyilli söylentide belki fazla Hester'ın derinden olmadığı Ve kadar şüphemizin de kabul modern etmeye daha o"
            },
            "vi": {
                "unshuffled": "Và chúng ta tất phải thuật lại rằng nó đã nung đốt thành dấu hằn vào ngực Hester sâu đến nỗi có lẽ trong lời đồn kia có nhiều phần sự thực hơn là đầu óc đa nghi của chúng ta trong thời hiện đại có thể sẵn sàng thừa nhận.",
                "shuffled": "ta phải thuật trong ta trong lẽ thể đại nỗi có nhận. nung đa hằn nghi đốt đồn lời vào dấu sâu Và hơn có sự hiện Hester của có phần thực kia ngực sẵn chúng tất thời nhiều sàng chúng đầu rằng đến là lại thừa đã óc nó thành"
            },
            "en": {
                "unshuffled": "And we must needs say, it seared Hester's bosom so deeply, that perhaps there was more truth in the rumor than our modern incredulity may be inclined to admit.",
                "shuffled": "admit. say, to inclined that the be more must so than it may needs modern we in rumor was deeply, incredulity perhaps our seared bosom there Hester's And truth"
            },
            "st": {
                "unshuffled": "Me re tlameha ho re, seared bosom ea Hester haholo, hoo mohlomong ho ne ho e-na le' nete ho feta menyenyetsi ho feta ho se lumele ha rona ea kajeno e ka ba tšekamelo ea ho lumela.",
                "shuffled": "se ho Me ea re ea ho kajeno hoo ea haholo, ho rona feta ho ho ho ne tšekamelo e lumela. feta ha seared e-na ka nete le' bosom re, mohlomong ho Hester tlameha ba lumele menyenyetsi"
            },
            "yo": {
                "unshuffled": "Àti pé a gbọ́dọ̀ nílò láti sọ pé, ó mú àyà Hester jinlẹ̀, pé bóyá òtítọ́ púpọ̀ wà nínú àròsọ ju àìgbàgbọ́ ìgbàlódé wa lọ lè fẹ́ láti gbà.",
                "shuffled": "pé láti wà àyà ìgbàlódé nílò púpọ̀ mú wa pé Àti pé, a lọ àròsọ ó gbà. láti fẹ́ Hester gbọ́dọ̀ òtítọ́ ju nínú jinlẹ̀, sọ bóyá lè àìgbàgbọ́"
            },
            "tn": {
                "unshuffled": "Mme re tshwanetse go re, re ne ra re, go ne go le thata gore re nne le tumelo ya ga Jehofa e e neng e le mo go yone, e ka tswa e le boammaaruri jo bogolo go feta tumelo ya rona ya gompieno.",
                "shuffled": "rona tumelo e yone, re e le e re le ga go gore go ne le mo tshwanetse ka boammaaruri e Mme nne re Jehofa ya gompieno. go jo e re, thata ne tswa ra bogolo re, neng go le feta ya ya go tumelo"
            },
            "ty": {
                "unshuffled": "E e tia ia tatou ia parau e, ua mauiui roa te ouma o Hester, e peneia'e ua rahi a'e te parau mau i roto i te parau i faahitihia i to tatou tiaturi ore no teie nei tau.",
                "shuffled": "roto mau i e, e tia teie ouma to tau. te o e parau faahitihia rahi i tatou ua te tiaturi te parau E peneia'e i no a'e tatou ua Hester, ia nei mauiui ia ore roa i parau"
            },
            "mai": {
                "unshuffled": "आ हमरासभकेँ ई कहबाक आवश्यकता अछि जे ई हेस्टरक छातीकेँ एतेक गहराई सँ प्रभावित कयलक, जे शायद अफवाहमे ओहिसँ बेसी सत्य छल जतेक हमर आधुनिक अविश्वास स्वीकार करय लेल इच्छुक भऽ सकैत अछि।",
                "shuffled": "जे हमरासभकेँ भऽ छल आ सकैत इच्छुक आवश्यकता हमर लेल ओहिसँ कयलक, जतेक सँ गहराई बेसी कहबाक करय जे स्वीकार एतेक अविश्वास शायद सत्य अफवाहमे ई अछि ई आधुनिक छातीकेँ प्रभावित अछि। हेस्टरक"
            },
            "mg": {
                "unshuffled": "Ary tsy maintsy mila miteny isika hoe, tena nampivoaka lalina ny tratran'i Hester izany, ka angamba nisy fahamarinana bebe kokoa tao anatin'ilay tsaho noho ny tsy finoana maoderina izay mety ho mora miaiky ny tsy finoana maoderina.",
                "shuffled": "izany, nisy mora tratran'i nampivoaka finoana ka izay mety kokoa isika tsy tao ny finoana maoderina miaiky tsy lalina hoe, ny Ary anatin'ilay ny bebe maoderina. tsy miteny Hester angamba fahamarinana noho mila maintsy ho tena tsaho"
            }
        }
        demo = demonstrations.get(lang, {}).get(mode, "")
        
        demo_passage = ""
        if prompt_setting != "zero-shot":
            demo_passage = f"""
            
            Here is an example:
            <passage>{demo}</passage>
            <output>"title": "The Scarlet Letter","author": "Nathaniel Hawthorne"</output>
            """

        #new code for clm
        if lang in ["st", "yo", "ty", "tn", "mai", "mg"]:
            prompt = f"""You are provided with a passage in {lang}. Your task is to carefully read the passage and determine which book this passage originates from and who the author is in English. You must make a guess IN ENGLISH, even if you are uncertain.
            {demo_passage}
            Here is the passage:
            <passage>{passage}</passage>

            Use the following format as output:
            <output>"title": "Book name","author": "Author name"</output>
        """
        else:
            prompt = f"""You are provided with a passage in {lang}. Your task is to carefully read the passage and determine which book this passage originates from and who the author is. You must make a guess, even if you are uncertain.
            {demo_passage}
            Here is the passage:
            <passage>{passage}</passage>
            Use the following format as output:
            <output>"title": "Book name","author": "Author name"</output>
            """
        return prompt
    except Exception as e:
        print(e)

def prepare_jsonl_input_file(csv_file_path, output_dir):
    """
    Reads the CSV file and creates a JSONL file with one API request per passage.
    Each request is given a unique custom_id in the format <column>_<row>.
    Saves the JSONL file in the output_dir using the book name.
    """
    df = pd.read_csv(csv_file_path)
    requests_list = []
    book_name = extract_book_name(csv_file_path)
    
    for col in df.columns:
        if col not in ALLOWED_COLUMNS:
            continue
        mode = "shuffled" if "shuffled" in col.lower() else "unshuffled"
        base_lang = col.split('_')[0]
        for idx, passage in df[col].items():
            prompt = construct_prompt(base_lang, passage, mode, "one-shot")
            custom_id = f"{col}_{idx}"
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100
                }
            }
            requests_list.append(request_obj)
    
    os.makedirs(output_dir, exist_ok=True)
    jsonl_file_path = os.path.join(output_dir, f"{book_name}_batch_input.jsonl")
    
    with open(jsonl_file_path, "w", encoding="utf-8") as f:
        for req in requests_list:
            f.write(json.dumps(req) + "\n")
    
    return requests_list, df, jsonl_file_path, book_name

def upload_file_to_openai(client, jsonl_file_path):
    """
    Uploads the JSONL file to OpenAI using the Files API.
    """
    with open(jsonl_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    return batch_file.id

def create_batch(client, input_file_id):
    """
    Initiates a batch process with the given input_file_id.
    """
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    return batch.id

def poll_batch_completion(client, batch_id, poll_interval=30):
    """
    Polls the batch status every poll_interval seconds until it is complete.
    Raises an exception if the batch fails or expires.
    """
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Batch {batch_id} status: {status}")
        if status == "completed":
            return batch
        elif status in ["failed", "expired"]:
            raise Exception(f"Batch {batch_id} failed or expired")
        time.sleep(poll_interval)

def download_and_parse_results(client, batch):
    """
    Downloads the output file from the completed batch and parses each JSON line.
    Returns a dictionary mapping custom_id to the model response.
    """
    output_file_id = batch.output_file_id
    file_response = client.files.content(output_file_id)
    file_contents = file_response.text

    results = {}
    for line in file_contents.splitlines():
        response_data = json.loads(line)
        custom_id = response_data.get("custom_id")
        if response_data.get("error"):
            results[custom_id] = None
        else:
            content = response_data["response"]["body"]["choices"][0]["message"]["content"]
            results[custom_id] = content
    return results

def update_dataset_with_results(df, results):
    """
    For each allowed column in the DataFrame, creates a new column that contains the response
    corresponding to each passage.
    """
    for col in df.columns:
        if col not in ALLOWED_COLUMNS:
            continue
        result_col = []
        for idx in df.index:
            custom_id = f"{col}_{idx}"
            result_col.append(results.get(custom_id))
        df[f"{col}_results"] = result_col
    return df

def process_csv(client, csv_file_path, batches_dir):
    """
    Process a single CSV: generate JSONL, upload file, create a batch, and return batch info.
    """
    requests_list, df, jsonl_file_path, book_name = prepare_jsonl_input_file(csv_file_path, batches_dir)
    print(f"Prepared {len(requests_list)} requests for {csv_file_path} in {jsonl_file_path}")
    
    input_file_id = upload_file_to_openai(client, jsonl_file_path)
    print(f"Uploaded file for {csv_file_path}. input_file_id: {input_file_id}")
    
    batch_id = create_batch(client, input_file_id)
    print(f"Created batch for {csv_file_path} with ID: {batch_id}")
    
    return batch_id, df, book_name

def extract_book_name(csv_file_path):
    """
    Extracts the book name from the CSV file name.
    If the file name contains "_name_cloze_", everything before that is returned.
    Otherwise, it returns the part before the first underscore.
    """
    base = os.path.basename(csv_file_path)
    if "_non_NE" in base:
        book_name = base.split("_non_NE")[0]
    else:
        book_name = base.split("_")[0]
    return book_name

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    
    base_dir = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/2024"
    all_csvs = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    csv_files = [f for f in all_csvs if f.endswith("_non_NE.csv")]    # Hardcoded folder path for storing batches and results
    batches_dir = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/direct_probing/batches/2024/non-ne-one-shot"
    
    if not csv_files:
        print("No CSV files found in", base_dir)
        return
    
    batch_info = {}
    
    for csv_file in csv_files:
        try:
            batch_id, df, book_name = process_csv(client, csv_file, batches_dir)
            batch_info[batch_id] = {"df": df, "book_name": book_name, "csv_file": csv_file}
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    #for every submitted batch, poll for completion and update the dataset
    for batch_id, info in batch_info.items():
        try:
            batch = poll_batch_completion(client, batch_id)
            results = download_and_parse_results(client, batch)
            updated_df = update_dataset_with_results(info["df"], results)
            updated_csv_path = os.path.join(batches_dir, f"{info['book_name']}_results.csv")
            updated_df.to_csv(updated_csv_path, index=False)
            print(f"Updated dataset saved to {updated_csv_path} for CSV {info['csv_file']}")
        except Exception as e:
            print(f"Error with batch {batch_id}: {e}")

if __name__ == "__main__":
    main()