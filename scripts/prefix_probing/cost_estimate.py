import pandas as pd
import tiktoken
import os


def count_token_in_string(string: str, encoding_name: str = "o200k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.

    Args:
    string (str): The text string to tokenize.
    encoding_name (str): The encoding name. Default is "o200k_base". New for gpt-4o.

    Returns:
    int: The number of tokens in the text string.
    """
    if isinstance(string, bytes):
        string = string.decode("utf-8", errors="ignore")  # Decode bytes to string

    # Load the encoding
    encoding = tiktoken.get_encoding(encoding_name)
    
    # Convert the string into tokens and count them
    num_tokens = len(encoding.encode(string))
    return num_tokens


def predict(lang, passage, bookname, word_count, prompt_setting="zero-shot"):
    # Demonstrations dictionary to provide an example passage and its continuation if not zero-shot
    demonstrations = {
        "es": {
            "first_half": "Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no",
            "second_half": "querer la vuelta de Jones. Cuando se les presentaba de esta forma, no sabían qué decir."
        },
        "tr": {
            "first_half": "Hayvanların en küçük bir kuşku duymadıkları tek bir",
            "second_half": "şey varsa, o da Jones'un geri dönmesini istemedikleriydi."
        },
        "vi": {
            "first_half": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ",
            "second_half": "vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán."
        },
        "en": {
            "first_half": "As Gregor Samsa awoke one morning from uneasy dreams",
            "second_half": "he found himself transformed in his bed into a gigantic insect."
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
        },
        "dv": {
            "unshuffled": "އެހެންވީމާ، އަޅުގަނޑުމެން ދަންނަވާލަން އެބަޖެހޭ، އެއީ ހެސްޓަރގެ ސިކުނޑި އެހާ ބޮޑަށް ކަނޑުވާލި ކަމެއް، އެއީ، އަޅުގަނޑުމެންގެ މިޒަމާނުގެ ނުތަނަވަސްކަން ޤަބޫލުކުރަން ޝައުޤުވެރިވާ ވަރަށްވުރެ، އެ ވާހަކަތަކުގައި ޙަޤީޤަތެއް އޮވެދާނެ.",
            "shuffled": "ޤަބޫލުކުރަން ކަމެއް، މިޒަމާނުގެ ހެސްޓަރގެ ޝައުޤުވެރިވާ އަޅުގަނޑުމެން އޮވެދާނެ. ނުތަނަވަސްކަން ޙަޤީޤަތެއް އަޅުގަނޑުމެންގެ އެބަޖެހޭ، އެހާ ދަންނަވާލަން ވަރަށްވުރެ، ވާހަކަތަކުގައި ކަނޑުވާލި އެއީ ސިކުނޑި އެހެންވީމާ، އެ ބޮޑަށް އެއީ،"
        }
    }

    demo_passage = ""
    # If prompt_setting is not zero-shot, include an example demonstration
    if prompt_setting != "zero-shot":
        demo = demonstrations.get(lang)
        if demo:
            demo_passage = f"""
            Here is an example:
            <passage>{demo['first_half']}</passage>
            <continuation>{demo['second_half']}</continuation>
            """

    text = f"""
       You are provided with an incomplete passage from a book. Your task is to continue the passage *in the SAME LANGUAGE* with {word_count}. Your response should ONLY be the continuation, directly quoted from the book. Do not repeat or rephrase the input.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <continuation>Passage continuation</continuation>
    # """
    # print(f"Type of passage: {type(passage)}")
    # print(passage)
    return count_token_in_string(text), count_token_in_string(passage)//2


def direct_probe(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)
        total_token = 0
        total_lines = 0
        total_output = 0

        for language in df.columns:
            if language not in ['Single_ent','line_no','Multiple_ents','masked','token_count'] and 'masked' not in language :
                for i in range(len(df)):
                    # print(language)
                    passage = df[language].iloc[i]
                    text, output = predict(language, passage, book_title, 50, prompt_setting=prompt_setting)
                    total_token += text
                    total_output += output
                    total_lines += 1
        
        avg_tokens_per_line = total_token / total_lines
        avg_tokens_per_output = round(total_output / total_lines)
        print(f"Total tokens in {book_title}: {total_token}")
        print(f"Average tokens per line in {book_title}: {round(avg_tokens_per_line)}")
        print(f'Projected token output per line in {book_title}: {avg_tokens_per_output}')
        print(f"Total lines in {book_title}: {total_lines}")
        print(f'Projected cost: {(total_token*5/1000000+(avg_tokens_per_output)*15/1000000)/2}')
        print('\n')
        return (total_token*3/1000000+(20)*3/1000000)
    except Exception as e:
        print(f"Error processing {book_title}: {e}")
        return 0

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    prompt_setting = "zero-shot"
    titles = get_folder_names('/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/')
    total = 0
    for title in titles:
        if title != 'raw':
            print(f'----------------- running {title} -----------------')
            total += direct_probe(f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/{title}/{title}_non_NE.csv", title, prompt_setting)
    print(f'\ntotal amount for all books: {total}')
