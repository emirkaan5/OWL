import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import re
from fireworks.client import Fireworks
import logging 
import sys
import io
import time  ### ADDED OR MODIFIED ###

client = Fireworks(api_key="")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('fireworks.client').setLevel(logging.WARNING)
logging.getLogger('fireworks').setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# Reconfigure stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def extract_output(text):
    soup = BeautifulSoup(text, 'html.parser')
    passages = soup.find_all('continuation')

    if passages:
        # Return the longest one if multiple <continuation> tags are found
        return max([passage.get_text() for passage in passages])
    else:
        return None


def predict(passage, lang, word_count, prompt_setting="zero-shot"):
    demonstrations = {
        "en": {
            "unshuffled": "A THRONG of bearded men, in sad-colored garments, and gray, steeple-crowned hats, intermixed with women, some wearing hoods and others bareheaded, was assembled in front of a wooden edifice, the door of which was heavily timbered with oak, and studded with iron spikes.",
            "shuffled": "of and with oak, heavily wearing intermixed wooden assembled of was timbered a hoods with was which edifice, hats, front and the A some men, steeple-crowned in sad-colored garments, of and studded iron in gray, spikes. bareheaded, THRONG bearded others door women, with"
        },
        "es": {
            "unshuffled": "UNA multitud de hombres barbudos, vestidos con trajes obscuros y sombreros de copa alta, casi puntiaguda, de color gris, mezclados con mujeres unas con caperuzas y otras con la cabeza descubierta, se hallaba congregada frente á un edificio de madera cuya pesada puerta de roble estaba tachonada con puntas de hierro.",
            "shuffled": "vestidos y unas copa multitud puntiaguda, madera congregada de roble con con de con de descubierta, frente caperuzas trajes con UNA alta, hallaba hierro. cabeza puerta la color con de tachonada de barbudos, puntas gris, otras se mujeres de un á sombreros pesada edificio y mezclados estaba cuya casi hombres obscuros"
        },
        "tr": {
            "unshuffled": "Saçları sakallı bir kalabalık, hüzünlü renkli elbiseler ve gri sivri tepeli şapkalardan oluşan, bazıları başörtüsü takan kadınlarla karışmış, demir çivilerle donatılmış, ağır meşe ağacından yapılmış kapısı olan bir ahşap binanın önünde toplanmıştı.",
            "shuffled": "bir renkli çivilerle karışmış, olan şapkalardan toplanmıştı. bir Saçları demir ahşap başörtüsü donatılmış, önünde takan kalabalık, hüzünlü elbiseler oluşan, sakallı binanın yapılmış meşe kapısı gri tepeli bazıları ağacından ve ağır sivri kadınlarla"
        },
        "vi": {
            "unshuffled": "Trước cửa một tòa nhà gỗ, cánh cửa bằng gỗ sồi nặng nề được đóng đinh sắt, một đám đông đàn ông râu ria, mặc quần áo màu buồn tẻ và đội mũ xám có chóp nhọn, xen lẫn với những người phụ nữ, một số đội mũ trùm đầu và những người khác để đầu trần, đã tụ tập lại.",
            "shuffled": "tẻ cửa được số trùm phụ nhà quần xen mặc chóp râu xám Trước tập màu khác những tụ đinh trần, nề tòa cửa đã gỗ ông đàn nặng với gỗ, người và đầu lẫn nhọn, đông buồn để sồi sắt, đội áo bằng đội một cánh ria, có một người mũ đầu mũ và lại. đám một nữ, đóng những"
        },
        "st": {
            "unshuffled": "Throng ea banna ba litelu, ka liaparo tse bohloko tse mebala-bala, le tse putsoa, likatiba tse roetsoeng moqhaka oa moepa, tse kopantsoeng le basali, ba bang ba apereng hood le ba bang bareheaded, ba ile ba bokana ka pel'a edifice ea lehong, monyako oa eona o neng o roaloa haholo ka oak, 'me ba studded ka li-spikes.",
            "shuffled": "o ba roetsoeng ba roaloa liaparo ka haholo moepa, li-spikes. oa oa lehong, 'me ba eona tse le tse le litelu, bang bohloko oak, banna studded o tse kopantsoeng ile Throng neng ba bareheaded, ba tse edifice tse mebala-bala, ba hood basali, bang ba pel'a likatiba ea bokana ea ka monyako apereng ka moqhaka putsoa, ka le"
        },
        "yo": {
            "unshuffled": "Ọ̀pọ̀lọpọ̀ àwọn ọkùnrin irùngbọ̀n, tí wọ́n wọ aṣọ aláwọ̀ ìbànújẹ́, àti àwọn fìlà aláwọ̀ eérú, tí wọ́n ní adé gíga, tí wọ́n so pọ̀ pẹ̀lú àwọn obìnrin, àwọn kan tí wọ́n wọ aṣọ ìbòjú àti àwọn mìíràn tí wọn kò ní orí, ni wọ́n kó jọ níwájú ilé igi, ìlẹ̀kùn èyí tí wọ́n fi igi oaku ṣe, tí wọ́n sì kún fún àwọn òpó irin.",
            "shuffled": "ìlẹ̀kùn wọ́n pọ̀ obìnrin, kò orí, igi, aṣọ aṣọ àwọn wọn kó wọ́n òpó tí wọ́n níwájú àwọn ilé jọ pẹ̀lú adé ní ní fún ìbòjú àwọn tí mìíràn àwọn fìlà àwọn Ọ̀pọ̀lọpọ̀ èyí wọ ìbànújẹ́, gíga, so fi oaku eérú, wọ irùngbọ̀n, ṣe, kan ni tí ọkùnrin tí àti wọ́n àti kún sì wọ́n àwọn wọ́n tí igi aláwọ̀ tí wọ́n aláwọ̀ irin. tí"
        },
        "tn": {
            "unshuffled": "23Ba ne ba apere diaparo tse di bogale, ba apere diaparo tse di bogale, le tse di rwesang ka thata, ba ba apereng dihempe le ba bangwe, ba phuthegile fa pele ga setlhare sa logong, mojako wa yona o o neng o rogwa thata ka eike, ba bo ba thubega ka ditshipi.",
            "shuffled": "o ne ba ba ga logong, dihempe thata, o bangwe, di ba ba rogwa ka ditshipi. diaparo neng ba thata di bo ba apere fa bogale, tse di le ka tse bogale, rwesang phuthegile eike, pele apere wa mojako o le apereng 23Ba tse thubega yona ba ka sa ba setlhare diaparo"
        },
        "ty": {
            "unshuffled": "UA PUTUPUTU mai te hoê pŭpŭ taata huruhuru taa, te mau ahu uo'uo, e te mau taupoo rehu e te taamu arapoa, e te tahi mau vahine, e te tahi pae ua ahuhia i te ahu uouo e te taamu arapoa, i mua i te hoê fare raau, ua î roa te opani i te raau, e ua î roa i te raau.",
            "shuffled": "e huruhuru te pae ua mai mau e î uo'uo, ua roa i te te hoê vahine, e UA te te roa raau. taamu mau mau te te î fare raau, te tahi tahi ahu hoê i opani i e ua te uouo arapoa, i ahu te te i e pŭpŭ ahuhia raau, taupoo rehu taamu arapoa, e mua taata te PUTUPUTU taa,"
        },
        "mai": {
            "unshuffled": "दाढ़ीवला पुरुषक भीड़, उदास रङ्गक वस्त्र आ धूसर, स्टीपल-मुकुटधारी टोपी, जे महिलासभक सङ्ग मिश्रित छल, किछु हुड पहिरने छल आ किछु नंगे माथ पर छल, एकटा लकड़ीक भवनक सोझाँ जमा कयल गेल छल, जकर दरवाजा ओकसँ भरल छल, आ लोहाक स्पाइकसँ जड़ल छल।",
            "shuffled": "कयल स्टीपल-मुकुटधारी पहिरने धूसर, छल, छल दरवाजा भवनक छल। जमा पर दाढ़ीवला लोहाक महिलासभक ओकसँ माथ मिश्रित आ नंगे हुड लकड़ीक जे आ छल, भीड़, जकर एकटा स्पाइकसँ भरल गेल किछु पुरुषक सङ्ग टोपी, वस्त्र सोझाँ रङ्गक उदास छल, किछु जड़ल आ छल,"
        },
        "mg": {
            "unshuffled": "Nivory teo anoloan'ny trano hazo iray ny lehilahy be volombava, nitafy akanjo miloko marevaka, ary satroka miloko volombatolalaka, nifangaro tamin'ny vehivavy, ny sasany nanao saron-doha ary ny hafa tsy nisaron-doha, dia nivory teo anoloan'ny trano hazo hazo, ny varavarana izay feno hazo oaka be dia be, ary feno tsipika vy.",
            "shuffled": "ny lehilahy ary ny be, hazo nitafy vehivavy, dia akanjo varavarana nanao hazo nisaron-doha, tsipika hazo teo tamin'ny iray sasany Nivory teo volombatolalaka, ny satroka anoloan'ny saron-doha miloko anoloan'ny ary hafa oaka nivory ny tsy trano dia be nifangaro trano miloko feno hazo, marevaka, ary volombava, vy. izay feno be"
        },
        "dv": {
            "unshuffled": "ހިތާމަވެރި ކުލައިގެ ހެދުންތަކާއި، ހުދުކުލައިގެ، ސްޓީޕަލް ތާޖުއަޅާފައިވާ ހެދުންއަޅައިގެން، އަންހެނުންނާ ގުޅިފައިވާ، ބައެއް މީހުން ހެދުން އަޅައިގެން، އަނެއްބައި މީހުންގެ އިސްތަށިގަނޑު ނިވާކޮށްގެން، ލަކުޑީގެ ޢިމާރާތެއްގެ ކުރިމައްޗަށް އެއްވެ، އެތަނުގެ ދޮރުގައި ވަރަށް ބޮޑަށް އޮށްޖަހާފައި، ދަގަނޑު ސްޕައިކްތަކުން ހަރުކޮށްފައި ހުއްޓެވެ.",
            "shuffled": "މީހުން ސްޕައިކްތަކުން ހުއްޓެވެ. ހެދުން ހިތާމަވެރި އެއްވެ، ނިވާކޮށްގެން، ޢިމާރާތެއްގެ ހުދުކުލައިގެ، ހަރުކޮށްފައި ދަގަނޑު ކުލައިގެ ގުޅިފައިވާ، އަންހެނުންނާ ތާޖުއަޅާފައިވާ އިސްތަށިގަނޑު ވަރަށް ކުރިމައްޗަށް ހެދުންތަކާއި، ބޮޑަށް ސްޓީޕަލް އެތަނުގެ އޮށްޖަހާފައި، ހެދުންއަޅައިގެން، މީހުންގެ އަނެއްބައި ދޮރުގައި ބައެއް ލަކުޑީގެ އަޅައިގެން،"
        }
    }

    demo = demonstrations.get(lang)
    
    # If your prompt_setting is not "0s", show an example. Otherwise, show nothing.
    demo_passage = ""
    if prompt_setting != "0s" and demo is not None:
        # Because the original code references 'first_half'/'second_half', we adapt:
        # (We can simply show 'unshuffled' and 'shuffled' as placeholders or skip)
        demo_passage = f"""
        Here is an example:
        <passage>{demo['unshuffled']}</passage>
        <continuation>{demo['shuffled']}</continuation>
        """

    prompt = f"""
       You are provided with an incomplete passage from a book. Your task is to continue the passage *in the SAME LANGUAGE* with {word_count}. Your response should ONLY be the continuation, directly quoted from the book. Do not repeat or rephrase the input.
        {demo_passage}

        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <continuation></continuation>
    """

    ### ADDED OR MODIFIED: Wrap the API call in try/except ###
    try:
        completion = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            max_tokens=100,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
    except Exception as e:
        logging.error(f"Error calling Fireworks API: {e}")
        time.sleep(1)  # Sleep for 1s
        return f"ERROR: {e}"

    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    else:
        logging.info(completion.choices[0].message.content)
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

        # Adjust the languages list to suit your needs
        languages = ["en", "vi", "es", "tr", "mg", "mai", "ty", "tn", "yo", "st"]
        for lang in languages:
            if lang in df.columns:
                logging.info(f"Running language: {lang}")
                output = []
                for i in range(len(df)):
                    full_passage = df[lang].iloc[i]
                    first_half, second_half, word_count = split_sentence_in_half(full_passage)
                    completion = predict(first_half, lang, word_count, prompt_setting)
                    trimmed_completion = remove_extra_suffix(
                        trim_common_prefix_suffix(first_half, completion),
                        len(second_half)
                    )
                    output.append([first_half, second_half, completion])
                
                output_df = pd.DataFrame(
                    output, 
                    columns=[
                        f'{lang}_first_half',
                        f'{lang}_second_half',
                        f'{lang}_Completion'
                    ]
                )
                df_out = pd.concat([df_out, output_df], axis=1)

        output_file_name = (
            f"/home/ekorukluoglu_umass_edu/beam2/BEAM/results/prefix_probe/"
            f"Llama-3.1-405b/2024/{book_title}_prefix_probe_Llama-3.1-405b_"
            f"{prompt_setting}_non_NE.csv"
        )
        df_out.to_csv(output_file_name, index=False, encoding='utf-8')
    except Exception as e:
        logging.error(e)


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names


if __name__ == "__main__":
    titles = get_folder_names('/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024')
    logging.info("Starting the prefix probe...")
    for title in titles:
        if "1984" not in title:
            logging.info(f"Processing {title} ...")
            prefixProbe(
                csv_file_name=f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{title}/{title}_non_NE.csv",
                book_title=title,
                prompt_setting="0s"
            )
            prefixProbe(
                csv_file_name=f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{title}/{title}_non_NE.csv",
                book_title=title,
                prompt_setting="1s"
            )
    logging.info("Done.")


#feb 26 46 dolar non ne
#feb  26 46 dolar unmasked
