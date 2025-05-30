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
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Lord Henry alzó las cejas y lo miró con asombro a través de las delgadas volutas de humo que, al salir de su cigarrillo con mezcla de opio, se retorcían adoptando extrañas formas.</passage>
            <answer>"title": "The Picture of Dorian Gray","author": "Oscar Wilde"</answer>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    elif lang == "tr":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Kendimden pek çok şey kattım buna.” Lord Henry divana boylu boyunca uzanarak güldü.</passage>
            <answer>"title": "The Picture of Dorian Gray","author": "Oscar Wilde"</answer>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    elif lang == "vi":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Oliver không hề thích thú kiểu đùa giỡn này và chuẩn bị thổ lộ sự bất bình của mình với hai anh bạn</passage>
            <answer>"title": "Oliver Twist","author": "Charles Dickens"</answer>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    else:
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>I am afraid I must be going, Basil.</passage>
            <answer>"title": "The Picture of Dorian Gray", "author": "Oscar Wilde"</answer>
            
            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
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
        print(f'Projected cost: {(total_token*3/1000000+(20)*3/1000000)}')
        print('\n')
        return (total_token*3/1000000+(20)*3/1000000)
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
    titles = get_folder_names('/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/')
    total = 0
    for title in titles:
        if title != 'raw':
            print(f'----------------- running {title} -----------------')
            total += direct_probe(f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/{title}/{title}_non_NE.csv")
    print(f'\ntotal amount for all books: {total}')