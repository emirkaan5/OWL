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
    # Load the encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Convert the string into tokens and count them
    num_tokens = len(encoding.encode(string))
    return num_tokens


# replace the prompt with the correct language when running experiments
def predict(lang, passage):
    if lang == "es":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.
    
            Example:
            <passage>Creo que incluso [MASK] estaría de acuerdo con tu opinión</passage>
            <name>Kafka</name>
    
            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows: 
            <name>Name</name>
        """
    elif lang == "tr":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example:
            <passage>Sadece [MASK] değil tüm grup hayduttu ve diğer insanlardan ayrı yaşıyorlardı</passage>
            <name>Robin</name>

            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows: 
            <name>Name</name>
            """
    elif lang == "vi":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.
    
            Example:
            <passage>"Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng [MASK] đang quá trông chờ vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán."</passage>
            <name>Alice</name>
            
            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows: 
            <name>Name</name>
        """
    else:
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example:
            <passage>Stay gold, [MASK], stay gold.</passage>
            <name>Ponyboy</name>
            
            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows: 
            <name>Name</name>
            """
    return count_token_in_string(text)


def direct_probe(csv_file_name):
    try:
        df = pd.read_csv(csv_file_name)
        total_token = 0
        total_lines = 0

        for language in df.columns:
            if language != 'Entity':
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    total_token += predict(language, passage)
                    total_lines += 1
        
        avg_tokens_per_line = total_token / total_lines
        print(f"Total tokens in {title}: {total_token}")
        print(f"Average tokens per line in {title}: {round(avg_tokens_per_line)}")
        print(f'Projected token output per line in {title}: {round(avg_tokens_per_line/2)}')
        print(f"Total lines in {title}: {total_lines}")
        print(f'Projected cost: {(total_token*5/1000000+(20)*15/1000000)}')
        print('\n')
        return (total_token*5/1000000+(20)*15/1000000)
    except:
        print(f'{title} is missing\n')
        return 0

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
    total = 0
    for title in titles:
        if title != 'raw':
            print(f'----------------- running {title} -----------------')
            total += direct_probe(f"/home/nhatminhle_umass_edu/Prompts/{title}/{title}_ner.csv")
    print(f'\ntotal amount for all books: {total}')
