import spacy
import os
import sys
import torch
from tqdm import tqdm
from datetime import datetime
import argparse
import json

from openai import OpenAI
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: This allows us to work with relative paths, but assumes that the script position in the repo remains the same!
os.chdir(sys.path[0])


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


def improved_prompt(text: str, openai_client: OpenAI, system_prompt: str) -> str:
    """
    Given a motion description, output enhanced motion description using OpenAI's GPT-3.5-turbo model.

    @param system_prompt:
    @param openai_client: OpenAI object for API calls
    @param text: motion description to refine
    @return: enhanced motion description
    """

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": system_prompt},
            {"role": "user",
             "content": text}
        ]
    )

    final_output = completion.choices[0].message.content.split("\n")[0]
    return text + " " + final_output


def text_enhancement(info_file_name: str, openai_client: OpenAI, target_folder: str, system_prompt: str) -> None:
    """
    Runs full text enhancement pipeline for all files specified by info_file_name and saves them at folder
    altered_texts.
    @param system_prompt:
    @param target_folder:
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
            file_path = os.path.join("../external_repos/momask-codes/dataset/HumanML3D/texts/", file_name + ".txt")

            # Open the file
            altered_text_path = os.path.join(target_folder, file_name + ".txt")
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

                        new_prompt = improved_prompt(text, openai_client, system_prompt)
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
            if line_count >= 10:
                break


def main():
    parser = argparse.ArgumentParser(description="Text Enhancement Pipeline")
    parser.add_argument("--folder_name", type=str, help="Specifies the target folder name where generated texts are saved to")
    parser.add_argument("--system_prompt", type=str, help="Name of JSON file containing system prompt",
                        default='extra_sentence.json')
    args = parser.parse_args()

    print(f"Using {DEVICE} device")

    client = OpenAI()

    if args.folder_name:
        target_folder = f"../prompt_enhancement/altered_texts/{args.folder_name}"
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        target_folder = f"../prompt_enhancement/altered_texts/altered_texts_{timestamp}"

    with open(f"../prompts/{args.system_prompt}", 'r') as file:
        system_prompt = json.load(file).get('system_prompt')

    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    text_enhancement("../external_repos/momask-codes/dataset/HumanML3D/test.txt", client, target_folder,
                     system_prompt)


if __name__ == "__main__":
    main()
