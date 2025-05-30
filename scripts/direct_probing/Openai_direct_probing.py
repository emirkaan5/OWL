from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os


client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)


def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    output = soup.find('output')
    if output:
        return output.decode_contents()
    
    return None


def predict(lang, passage, mode, prompt_setting):
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
            
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",

            max_tokens=100,

            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        extract = extract_output(completion.choices[0].message.content)
        return extract if extract else completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing passage: {e}")
        return None


def direct_probe(csv_file_name, book_title, prompt_setting):
    try:
        df = pd.read_csv(csv_file_name)
        
        # ensuring only specified columns are run
        allowed_columns = [
            "en", "es", "tr", "vi", "en_shuffled", "es_shuffled", "tr_shuffled", "vi_shuffled",
            "st", "yo", "ty", "tn", "mai", "mg", "st_shuffled", "yo_shuffled", "ty_shuffled", "tn_shuffled", "mai_shuffled", "mg_shuffled"
        ]

        for language in df.columns:
            if language not in allowed_columns:
                continue
            
            print(f"Processing column: {language}")
            output = []
            mode = "shuffled" if "shuffled" in language.lower() else "unshuffled"
            
            for i in range(len(df)):
                passage = df[language].iloc[i]
                try:
                    base_language = language.split('_')[0]
                    content = predict(base_language, passage, mode, prompt_setting)
                    output.append(content)
                    print(f'Row {i}: {content}')
                except Exception as e:
                    print(f"\nERROR")
                    print(f"Error at row {i} for column {language}: {e}\n")
                    output.append(None)  #NOTE: to maintain DataFrame structure in case of error
            
            index_of_language = df.columns.get_loc(language)
            output_col = pd.Series(output)
            df.insert(index_of_language + 1, f"{language}_results", output_col)

        output_file_name = f"/Users/alishasrivastava/BEAM-scripts/BEAM/results/direct_probe/gpt-4o-2024-11-20/non_ne_zero_shot/{book_title}_direct_probe_gpt-4o-2024-11-20_{prompt_setting}.csv"
        df.to_csv(output_file_name, index=False, encoding='utf-8')
        print(f"Results saved to {output_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names


if __name__ == "__main__":
    titles = get_folder_names('/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts')
    titles.remove("2024")
    for title in titles:
        print(f'----------------- running {title} -----------------')
        direct_probe(csv_file_name=f"/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/{title}/{title}_non_NE.csv", book_title=title, prompt_setting="zero-shot") # modify the prompt setting here
            