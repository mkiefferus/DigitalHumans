from openai import OpenAI
import os
import spacy
import json
from tqdm import tqdm
import torch
import sys
import argparse
from datetime import datetime
import numpy as np
from utils.utils import ROOT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")
# Note: This allows us to work with relative paths, but assumes that the script position in the repo remains the same!
os.chdir(sys.path[0])

def _annotate_motion(motion, nlp):
    """Add part of speech tags to motion description."""
    motion = motion.replace('-', '')
    doc = nlp(motion)
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

def export_data(data:json, annotations_dict:dict[str:list[str]], output_folder:str):
    """Generate dataset with refined text"""
    
    try: 
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Could not locate en_core_web_sm model, please install with \"python -m spacy download en_core_web_sm\"")
        exit(1)

    for file_name, output in data.items():

        # Check if annotations exist for the filename
        if file_name not in annotations_dict:
            raise KeyError(f"Annotations missing for file: {file_name}")
        
        altered_text_path = os.path.join(output_folder, file_name + ".txt")
        open(altered_text_path, 'w').close()  # Clear the file

        with open(altered_text_path, 'a') as altered_file:
            
            for motion, annotation in zip(output, annotations_dict[file_name]):
                
                # Add part of speech tags to motion description
                word_list, pose_list = _annotate_motion(motion, nlp)
                motion_tag = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])

                # Write final refined text to file
                altered_file.write(motion + '#' + motion_tag + '#' + annotation + '\n')


def process_data(filenames:list[str]) -> tuple[str, dict[str, list[str]]]:
    """Process files into JSON format and extract annotations.

    Args:
        filenames (List[str]): List of filenames to be processed.

    Returns:
        Tuple[str, Dict[str, List[str]]]: A tuple containing the JSON string of motions and a dictionary of annotations.
    """

    base_dir = os.path.join(HUMAN_ML_DIR, "texts")

    input_dict = {}
    annotations_dict = {}

    for filename in filenames:

        file_path = os.path.join(base_dir, filename + ".txt")
        
        # Initialize lists for each filename
        input_dict[filename] = []
        annotations_dict[filename] = []

        try:
            with open(file_path, 'r') as opened_file:
                lines = opened_file.readlines()
            
            for line in lines:
                content, _, a1, a2 = line.strip().rsplit("#", 3)

                # Append content and annotations to respective lists
                input_dict[filename].append(content)
                annotations_dict[filename].append(f"#{a1}#{a2}")

        # Handle exceptions
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except Exception as e:
            raise Exception(f"An error occurred while processing {file_path}: {str(e)}")
        
    # Convert input dictionary to JSON format
    json_input = json.dumps(input_dict, indent=4)

    return json_input, annotations_dict


def get_text_refinement(data:json, system_prompt:str, model:str, client) -> json:
    """Use OpenAI API for text refinement."""

    batch_prompt = """You are a book author known for your detailed motion descriptions and simple vocabulary and are given a JSON of format: 
        "filename1": [
        "motion1",
        "motion2",
        "motion3",
        ],
        "filename2": [
        "motion4",
        "motion5",
        ]
        and so on. 
        You output in the JSON format. Your answer will be in the same string, no extra strings.
        You will: keep the order of the motions, elaborate each motion but stay concise, treat each motion as a separate task and go through all of them, only focus on the motion description.
        You will NOT: explain if it is the first or second motion, skip motions.
        """
    
    new_system_prompt = batch_prompt + system_prompt

    new_prompt = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": new_system_prompt},
            {"role": "user", "content": f"list of strings: {data}"}
            ]
        )
    return json.loads(new_prompt.choices[0].message.content)


def refine_text(data_folder:str, 
                output_folder:str, 
                system_prompt:str, 
                batch_size:int=3, 
                model:str="gpt-3.5-turbo", 
                client=OpenAI(), 
                refine_specific_samples_txt_path=None,
                stop_after_n_samples=np.inf):
    """Refines text in datafolder using given model and system prompt"""

    # Get list of filenames to process (without .txt)
    if refine_specific_samples_txt_path is not None:
        with open(refine_specific_samples_txt_path, 'r') as file:
            file_names = file.read().splitlines()
        files = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.txt') and f.split('.')[0] in file_names]
    else:
        files = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.txt')]
    print(f"Total files found: {len(files)}")

    # number of batches
    num_batches = (len(files) + batch_size - 1) // batch_size  # This ensures all files are included even if the last batch is smaller

    progress = tqdm(range(num_batches), desc="Processing batches")

    for i in progress:
        batch = files[i*batch_size:(i+1)*batch_size]

        try:
            input, annotations = process_data(batch)
            data = get_text_refinement(data=input, system_prompt=system_prompt, model=model, client=client)
            export_data(data=data, annotations_dict=annotations, output_folder=output_folder)

        except Exception as e:
            print(f"An error occurred while processing batch {i + 1} ({batch}): {e}")

        if i == stop_after_n_samples -1:
            break

        progress.update(1)
    progress.close()

    print("Text refinement complete.")


def main():
    parser = argparse.ArgumentParser(description="Text Enhancement Pipeline")
    parser.add_argument("--folder_name", type=str,
                        help="Specifies the target folder name where generated texts are saved to")
    parser.add_argument("--system_prompt", type=str, help="Name of JSON file containing system prompt",
                        default='extra_sentence.json')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for text enhancement")
    parser.add_argument("--early_stop", type=int, default=np.inf, help="Stop after n refined samples for testing purposes")
    args = parser.parse_args()

    print(f"Using {DEVICE} device")

    client = OpenAI()

    if args.folder_name:
        target_folder = os.path.join(HUMAN_ML_DIR, args.folder_name)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        target_folder = os.path.join(HUMAN_ML_DIR, f"altered_texts_{timestamp}")

    with open(f"../prompts/{args.system_prompt}", 'r') as file:
        system_prompt = json.load(file).get('system_prompt')

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    refine_text(data_folder="../external_repos/momask-codes/dataset/HumanML3D/texts/", 
            output_folder=target_folder,
            system_prompt=system_prompt,
            batch_size=args.batch_size,
            client=client,
            refine_specific_samples_txt_path="../external_repos/momask-codes/dataset/HumanML3D/test.txt",
            stop_after_n_samples=args.early_stop
        )



if __name__ == "__main__":
    main()
