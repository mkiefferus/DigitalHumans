import spacy
import os
import torch
from tqdm import tqdm

from openai import OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_text(sentence, nlp):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def improved_prompt(text: str, openai_client: OpenAI) -> str:
    """
    Given a motion description, output enhanced motion description using OpenAI's GPT-3.5-turbo model.

    @param openai_client: OpenAI object for API calls
    @param text: motion description to refine
    @return: enhanced motion description
    """

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You will receive a motion description. Answer with one short sentence that precisely describes the movement of the most relevant body part for this motion. Here is an example: a man kicks something or someone with his left leg Your output should be: They do so by rapidly lifting and extending their left leg."},
            {"role": "user",
             "content": text}
        ]
    )

    # print(completion.choices[0].message.content)
    final_output = completion.choices[0].message.content.split("\n")[0]
    return text + " " + final_output


def text_enhancement(info_file_name: str, openai_client: OpenAI) -> None:
    """
    Runs full text enhancement pipeline for all files specified by info_file_name and saves them at folder
    altered_texts.
    @param openai_client: OpenAI object for API calls
    @param info_file_name: Text file specifying which files belong to the dataset split
    """
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Could not locate en_core_web_sm model, please install with \"python -m spacy download en_core_web_sm\"")
        exit(1)

    num_lines = sum(1 for line in open(info_file_name))
    with open(info_file_name, 'r') as file:
        line_count = 0
        for line in tqdm(file, total=num_lines, desc="Generating enhanced motion descriptions"):
            # Use this if generating crashed and you want to resume somewhere
            # if line_count < 1517:
            #     line_count += 1
            #     continue
            # Remove newline character and any leading/trailing whitespace
            file_name = line.strip()

            # Construct the full path to the file
            file_path = os.path.join("texts/", file_name + ".txt")

            # Open the file
            altered_text_path = os.path.join("altered_texts/", file_name + ".txt")
            try:
                with open(file_path, 'r') as opened_file:
                    # Clear the file
                    open(altered_text_path, 'w').close()
                    # Read the file line by line
                    for prompt_line in opened_file:
                        # Split the line by '#' to separate text and annotations
                        parts = prompt_line.strip().split('#')
                        # Extract the text part generate new prompt and part-of-speech tagging
                        text = parts[0].strip()

                        new_prompt = improved_prompt(text, openai_client)
                        word_list, pose_list = process_text(new_prompt, nlp)
                        new_prompt_tag = ' '.join(
                            ['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
                        # Extract annotations
                        annotations = "#" + parts[2].strip().split('#')[0].strip() + "#" + parts[3].strip().split('#')[
                            0].strip()
                        with open(altered_text_path, 'a') as altered_file:
                            altered_file.write(new_prompt + '#' + new_prompt_tag + annotations + '\n')
            except UnicodeDecodeError:
                # writing code
                print(f"Error decoding file {file_path}")
                if os.path.exists(altered_text_path):
                    # Delete the file (assumes we are replacing files in HumanML3D, i.e. will just result in unaltered file being used)
                    os.remove(altered_text_path)

            line_count += 1
            # For debugging purposes
            # if line_count >= 10:
            #     break


def main():
    print(f"Using {DEVICE} device")

    client = OpenAI()
    text_enhancement("../../Similarity_Search/test.txt", client)


if __name__ == "__main__":
    main()
