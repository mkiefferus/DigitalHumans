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
import yaml
from utils.utils import ROOT_DIR, MOMASK_REPO_DIR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")
# Note: This allows us to work with relative paths, but assumes that the script position in the repo remains the same!
os.chdir(sys.path[0])

def _continue_folder(continue_folder_path:str, data_folder_path:str, refine_specific_samples_txt_path:str) -> list[str]:
    """(Helperfunction) Continue refining text at checkpoint"""

    # Check if path exists
    if not os.path.isdir(continue_folder_path):
        raise NotADirectoryError(f"The specified path '{continue_folder_path}' is not a directory or does not exist.")

    # Load the names of all files in that folder that end with .txt
    continue_file_names = {file_name.split('.')[0] for file_name in os.listdir(continue_folder_path) if file_name.endswith(".txt")}
    data_file_names = {file_name.split('.')[0] for file_name in os.listdir(data_folder_path) if file_name.endswith(".txt")}

    remaining_files = list(data_file_names - continue_file_names)

    # If a refine_specific_samples_txt_path is provided, only process the files in that list
    if refine_specific_samples_txt_path is not None:
        with open(refine_specific_samples_txt_path, 'r') as file:
            file_names = file.read().splitlines()
        remaining_files = [f for f in remaining_files if f in file_names]

    print(f"Continuing from folder: {continue_folder_path} \n -> {len(remaining_files)-len(continue_file_names)}/{len(remaining_files)} files remaining.")

    return remaining_files

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

def export_data(data:json, annotations_dict, output_folder:str):
    """Generate dataset with refined text"""
    
    try: 
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Could not locate en_core_web_sm model, please install with \"python -m spacy download en_core_web_sm\"")
        exit(1)


    for file_name, motions in data.items():

        # Check if annotations exist for the filename
        if file_name not in annotations_dict:
            raise KeyError(f"Annotations missing for file: {file_name}")
        
        altered_text_path = os.path.join(output_folder, file_name + ".txt")
        with open(altered_text_path, 'w') as altered_file:
            
            for motion_key, content in motions.items():
                annotation = annotations_dict[file_name][motion_key] if motion_key in annotations_dict[file_name] else 'No annotation'

                # Add part of speech tags to motion description
                doc = nlp(content)
                motion_tag = ' '.join([f"{token.text}/{token.pos_}" for token in doc])

                # Write final refined text to file
                altered_file.write(f"{content}#{motion_tag}#{annotation}\n")


def process_data(filenames:list):
    """Process files into JSON format and extract annotations.

    Args:
        filenames (List[str]): List of filenames to be processed.

    Returns:
        Tuple[str, Dict[str, Dict[str]]]: A tuple containing the JSON string of motions and a dictionary of annotations.
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
            
            lines_dict = {}
            for i, line in enumerate(lines):
                content, _, a1, a2 = line.strip().rsplit("#", 3)

                # Append content and annotations to respective lists
                lines_dict[f"motion{i+1}"] = content
                annotations_dict[filename].append(f"{a1}#{a2}")

            input_dict[filename] = lines_dict

        # Handle exceptions
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except Exception as e:
            raise Exception(f"An error occurred while processing {file_path}: {str(e)}")
        
    # Convert input dictionary to JSON format
    json_input = json.dumps(input_dict, indent=4)

    return json_input, annotations_dict


def get_text_refinement(data, system_prompt:str, example_prompt, model:str, client, BATCH_PROCESSING:bool=False) -> json:
    """Use OpenAI API for text refinement."""

    if BATCH_PROCESSING:
        batch_prompt = """You are a book author known for your detailed motion descriptions and simple vocabulary. Your task is to generate descriptions for a given list of motions represented in a JSON format. 
        Each motion is associated with a specific filename and is identified uniquely. The descriptions should focus solely on the motion itself and avoid mentioning body parts unless integral to the motion.

            Format of input JSON:
            {
                "filename1": {
                    "motion1": "brief description",
                    "motion2": "brief description",
                    "motion3": "brief description"
                },
                "filename2": {
                    "motion4": "brief description",
                    "motion5": "brief description"
                }
                // More files and motions can follow the same pattern.
            }

            Required output format:
            Your output must also be in JSON format. Each motion description must be elaborate, maintaining the order of the motions as presented in the input. 
            Do not skip any motions or include descriptions of muscle details. 
            Each motion should be described in one or two sentences that elaborate on the brief description, without changing the nature of the motion described.

            Example of an optimal output:
            {
                "filename1": {
                    "motion1": "The torso sways slightly to the left, while the arms remain still. The legs move in response to maintain balance.",
                    "motion2": "The legs move sideways to the left and then to the right, with minimal movement from the upper body.",
                    "motion3": "The legs move in a fluid motion, stepping to the right and crossing the left foot behind the right, before returning to the starting position."
                },
                "filename2": {
                    "motion1": "The entire body bounces up and down as the figure performs jumping jacks, with arms moving up and out.",
                    "motion2": "The torso bobs up and down three times as the man does jumping jacks, with arms extended and legs moving in a small circle.",
                    "motion3": "The person's entire body moves in a fluid motion, bouncing up and down while performing jumping jacks."
                }
            }
        """

        # Load example prompts for assistant and user
        example_prompt = example_prompt.get('batch')
        ex_user = json.dumps(example_prompt.get('user'), indent=4)
        ex_assistant = json.dumps(example_prompt.get('assistant'), indent=4)

    # Account for single file processing
    else:
        batch_prompt = """You are a book author known for your detailed motion descriptions and simple vocabulary. You receive an instruction and a sentence. 
        Your task is to generate a detailed description of the motion in the sentence. The description should focus solely on the motion itself and avoid mentioning body parts. Your answer is one, max two sentences. It must be below 70tokens/~60 words.
        """
        example_prompt = example_prompt.get('single')
        ex_user = json.dumps(example_prompt.get('user'))
        ex_assistant = json.dumps(example_prompt.get('assistant'))

    new_system_prompt = batch_prompt + system_prompt


    new_prompt = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", 
             "content": new_system_prompt},
            {"role": "user", 
             "content": ex_user},
            {"role": "assistant", 
             "content": ex_assistant},
            {"role": "user",
             "content": data}
            ]
        )
    
    refined_text = new_prompt.choices[0].message.content
    
    if BATCH_PROCESSING:
        # Cut everything before the first '{' and after the last '}'
        refined_text = refined_text[refined_text.find('{'):refined_text.rfind('}')+1]
        
        return json.loads(refined_text)
    
    else:
        return refined_text


def refine_text(data_folder:str, 
                output_folder:str, 
                system_prompt:str, 
                example_prompt,
                batch_size:int=3, 
                model:str="gpt-3.5-turbo", 
                client=OpenAI(), 
                refine_specific_samples_txt_path=None,
                stop_after_n_batches=np.inf,
                continue_previous=None):
    """Refines text in datafolder using given model and system prompt"""

    if continue_previous is not None:
        files = _continue_folder(continue_previous, data_folder, refine_specific_samples_txt_path)
        output_folder = continue_previous

    else:
        # Get list of filenames to process (without .txt)
        if refine_specific_samples_txt_path is not None:
            with open(refine_specific_samples_txt_path, 'r') as file:
                file_names = file.read().splitlines()
            files = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.txt') and f.split('.')[0] in file_names]
        else:
            files = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.txt')]

    print(f"Total files found: {len(files)}")

    BATCH_PROCESSING = False if batch_size < 1 else True



    if BATCH_PROCESSING:

        # number of batches
        num_batches = (len(files) + batch_size - 1) // batch_size  # This ensures all files are included even if the last batch is smaller

        if refine_specific_samples_txt_path is not None:
            num_batches = min(num_batches, stop_after_n_batches)

        progress = tqdm(range(num_batches), desc="Processing batches")

        for i in progress:
            batch = files[i*batch_size:(i+1)*batch_size]

            try:
                input, annotations = process_data(batch) # BATCH_PROCESSING is False
                data = get_text_refinement(data=input, system_prompt=system_prompt, example_prompt=example_prompt, model=model, client=client, BATCH_PROCESSING=BATCH_PROCESSING)
                export_data(data=data, annotations_dict=annotations, output_folder=output_folder)

            except Exception as e:
                print(f"An error occurred while processing batch {i + 1} ({batch}): {e}")

            if i == stop_after_n_batches -1:
                break

            progress.update(1)
        progress.close()

    # Process files one by one
    else:
        files = files[:min(stop_after_n_batches, len(files))]

        for file in tqdm(files, desc=f"Processing files"):

            try:
                input, annotations = process_data([file]) # BATCH_PROCESSING is True

                # Convert input and annotions to dict
                input = json.loads(input)

                for motion, desc in input[file].items():

                    refined_text = get_text_refinement(data=desc, system_prompt=system_prompt, example_prompt=example_prompt, model=model, client=client, BATCH_PROCESSING=BATCH_PROCESSING)
                    input[file][motion] = refined_text
                    export_data(data=input, annotations_dict=annotations, output_folder=output_folder)

            except Exception as e:
                    print(f"An error occurred while processing file {file}: {e}")


    print("Text refinement complete.")


def main():
    parser = argparse.ArgumentParser(description="Text Enhancement Pipeline")
    parser.add_argument("--folder_name", type=str,
                        help="Specifies the target folder name where generated texts are saved to")
    parser.add_argument("--system_prompt", type=str, help="Name of JSON file containing system prompt",
                        default='extra_sentence.json')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for text enhancement")
    parser.add_argument("--early_stop", type=int, default=np.inf, help="Stop after n refined samples for testing purposes")
    parser.add_argument("--continue_previous", type=str, default=None, help="Continue refining texts from a specific folder")
    parser.add_argument("--refine_all_samples", type=bool, default=False, help="Refine only all samples. Default: refine test samples only")
    parser.add_argument("--from_config", type=bool, default=False, help="Load configuration from config.yaml")
    args = parser.parse_args()

    client = OpenAI()

    # Load args from config file if 'from_config'
    if args.from_config:
        with open(os.path.join(ROOT_DIR, "prompt_enhancement_models", "config.yaml"), 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        print("Overwriting args with config file...")

        # Overwrite args with the ones from the config file
        for arg, value in config.items():
            setattr(args, arg, value)

        # Set client and model
        base_url = getattr(args, 'base_url', False)
        api_key = getattr(args, 'api_key', False)

        if base_url and api_key:
            client = OpenAI(
                base_url = base_url,
                api_key=api_key
            )

    print(f"Using {DEVICE} device")

    if args.early_stop == -1: # config file does not support np.inf
        args.early_stop = np.inf

    model = getattr(args, 'model', 'gpt-3.5-turbo')


    # Ensure folder structure exists
    if args.folder_name:
        target_folder = os.path.join(HUMAN_ML_DIR, args.folder_name)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        target_folder = os.path.join(HUMAN_ML_DIR, f"altered_texts_{timestamp}")
        
    target_folder = args.continue_previous if args.continue_previous is not None else target_folder

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    # Load example prompt for model assistant and user
    with open(os.path.join(ROOT_DIR, "prompts_examples" ,f"ex_{args.system_prompt}"), 'r') as file:
        example_prompt = json.load(file)

    # Load system prompt
    with open(f"prompts/{args.system_prompt}", 'r') as file:
        system_prompt = json.load(file).get('system_prompt')

    if not args.refine_all_samples:
        refine_specific_samples_txt_path = os.path.join(MOMASK_REPO_DIR, "dataset", "HumanML3D", "test.txt")

    _config = {
        "config": {
            "folder_name": target_folder,
            "system_prompt": args.system_prompt,
            "client": str(client),
            "model": model,
            "batch_size": args.batch_size,
            "early_stop": args.early_stop,
            "continue_previous": args.continue_previous,
            "refine_specific_samples_txt_path": refine_specific_samples_txt_path
        }
    }
    print("Configuration: ", _config)
    
    # Write configuration to a YAML file
    config_path = os.path.join(target_folder, 'config.yaml')
    with open(config_path, 'w') as yaml_file:
        yaml.dump(_config, yaml_file, default_flow_style=False)


    refine_text(data_folder="external_repos/momask-codes/dataset/HumanML3D/texts/", 
            output_folder=target_folder,
            system_prompt=system_prompt,
            example_prompt=example_prompt,
            batch_size=args.batch_size,
            client=client,
            model=model,
            refine_specific_samples_txt_path=refine_specific_samples_txt_path,
            stop_after_n_batches=args.early_stop,
            continue_previous=args.continue_previous
        )


if __name__ == "__main__":
    main()
