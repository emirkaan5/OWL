import os
import glob
import json
import time
import openai
import pandas as pd

ALLOWED_COLUMNS = [
    "st_shuffled", "yo_shuffled", "ty_shuffled", "tn_shuffled", "mai_shuffled", "mg_shuffled"
]

def construct_prompt(lang, passage, mode, prompt_setting="zero-shot"):
    try:
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
        },
        "st": {
            "unshuffled": "'Me re tlameha ho re, earared [MASK] bosom e tebileng haholo, hore mohlomong ho na le 'nete e ngata ka menyenyetsi ho feta ho se lumele ha rona ea kajeno e ka ba tšekamelo ea ho lumela.",
            "shuffled": "earared e re 'Me kajeno e re, ho lumele ea mohlomong e hore menyenyetsi ngata ha ka rona 'nete ba na ka ea le haholo, tšekamelo feta ho tebileng se bosom ho ho lumela. ho tlameha [MASK]"
        },
        "yo": {
            "unshuffled": "Àti pé a gbọ́dọ̀ nílò láti sọ pé, ó mú àyà [MASK] jinlẹ̀, pé bóyá òtítọ́ púpọ̀ wà nínú àròsọ ju àìgbàgbọ́ ìgbàlódé wa lọ lè fẹ́ láti gbà.",
            "shuffled": "nínú a jinlẹ̀, ìgbàlódé [MASK] òtítọ́ ó púpọ̀ pé, pé mú wà láti lọ àròsọ àìgbàgbọ́ sọ wa láti àyà gbọ́dọ̀ lè fẹ́ pé nílò ju Àti gbà. bóyá"
        },
        "tn": {
            "unshuffled": "Mme re tshwanetse ra re, [MASK] le fa go ntse jalo, go ne go na le boammaaruri jo bogolo go feta mo tumelong ya rona ya gompieno.",
            "shuffled": "tumelong le gompieno. re, jalo, mo ya ne feta jo bogolo boammaaruri go go le [MASK] ra ya ntse go na fa Mme rona tshwanetse go re"
        },
        "ty": {
            "unshuffled": "E e ti'a ia tatou ia parau e, ua î roa te ouma o [MASK] i te reira, e peneia'e ua rahi a'e te parau mau i roto i te parau i to tatou ti'aturi-ore-raa no teie tau.",
            "shuffled": "tau. mau E te tatou ti'a teie e parau rahi te e, parau î i ua ia reira, [MASK] te to tatou i parau ua ti'aturi-ore-raa ouma roto te roa i peneia'e a'e ia e no o i"
        },
        "mai": {
            "unshuffled": "आ हमरासभकेँ ई कहबाक आवश्यकता अछि जे ई [MASK] छातीकेँ एतेक गहराई सँ प्रभावित कयलक, जे शायद अफवाहमे ओहिसँ बेसी सत्य छल जतेक हमर आधुनिक अविश्वास स्वीकार करय लेल इच्छुक भऽ सकैत अछि।",
            "shuffled": "अछि आवश्यकता अफवाहमे प्रभावित गहराई [MASK] सँ हमर ओहिसँ अछि। ई लेल बेसी छल जे जतेक हमरासभकेँ अविश्वास करय आ कयलक, स्वीकार कहबाक आधुनिक ई छातीकेँ शायद सकैत एतेक इच्छुक सत्य जे भऽ"
        },
        "mg": {
            "unshuffled": "Ary tsy maintsy mila miteny isika hoe, nampivoaka lalina ny tratran'i [MASK] izany, ka angamba nisy fahamarinana bebe kokoa tao anatin'ilay tsaho fa tsy mety ho mora miaiky ny tsy finoana maoderina ananantsika.",
            "shuffled": "miteny tsy miaiky mora maintsy mila ny tsy kokoa [MASK] tsy fa ka bebe mety tratran'i izany, anatin'ilay fahamarinana ho maoderina lalina tsaho finoana angamba Ary isika tao ananantsika. hoe, nisy nampivoaka ny"
        }
        }
        demo = demonstrations.get(lang, {}).get(mode, "")
        
        demo_passage = ""
        if prompt_setting != "zero-shot":
            demo_passage = f"""
            Here is an example:
            <passage>{demo}</passage>
            <output>Hester</output>
            """

        #new code for clm
        if lang in ["st", "yo", "ty", "tn", "mai", "mg"]:
            prompt = f"""You are provided with a passage in {lang}. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess IN ENGLISH, even if you are uncertain.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
        <output>Name</output>
       """
        else:
            prompt = f"""You are provided with a passage in {lang}. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.
            {demo_passage}
            Here is the passage:
            <passage>{passage}</passage>

            Use the following format as output:
            <output>Name</output>
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
    if "_masked_passages" in base:
        book_name = base.split("_masked_passages")[0]
    else:
        book_name = base.split("_")[0]
    return book_name

def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
    
    base_dir = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/Alice_in_Wonderland"
    all_csvs = glob.glob(os.path.join(base_dir, "**/*.csv"), recursive=True)
    csv_files = [f for f in all_csvs if "2024" not in f and f.endswith("_masked_passages.csv")]    # Hardcoded folder path for storing batches and results
    batches_dir = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/name_cloze_task/batches/alice"
    
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