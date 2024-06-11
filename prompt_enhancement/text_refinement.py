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

# Note: This allows us to work with relative paths, but assumes that the script position in the repo remains the same!
os.chdir(sys.path[0])

def _continue_folder(continue_folder_path:str, data_folder_path:str, refine_specific_samples_txt_path:str):
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
                # print(motion_key)
                # print(content)
                annotation = annotations_dict[file_name][motion_key] if motion_key in annotations_dict[file_name] else '0.0#0.0'
                # print(annotation)
                # print("")
                
                # Add part of speech tags to motion description
                doc = nlp(content)
                motion_tag = ' '.join([f"{token.text}/{token.pos_}" for token in doc])

                # Write final refined text to file
                altered_file.write(f"{content}#{motion_tag}#{annotation}\n")


def process_data(filenames:list, observation_instead_of_motion=False):
    """Process files into JSON format and extract annotations.

    Args:
        filenames (List[str]): List of filenames to be processed.
        observation_instead_of_motion (bool): if True, the function will put out a dictionary with key "observationX" instead of "motionX". Default is False.

    Returns:
        Tuple[str, Dict[str, Dict[str]]]: A tuple containing the JSON string of motions and a dictionary of annotations.
    """

    base_dir = TEXTS_DIR

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
                if observation_instead_of_motion:
                    lines_dict[f"observation{i+1}"] = content
                else:
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


def get_text_refinement(data, system_prompt:str, example_prompt, model:str, client, BATCH_PROCESSING:bool=False, use_cross_sample_information:bool=False) -> json:
    """Use OpenAI API for text refinement."""

    messages = []
    
    if BATCH_PROCESSING:
        batch_prompt = """You are an average human known for your detailed motion descriptions and simple vocabulary. Your task is to generate descriptions for a given list of motions represented in a JSON format. 
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
        """

        # Load example prompts for assistant and user
        if example_prompt:
            example_prompt = example_prompt.get('batch')
            ex_user = json.dumps(example_prompt.get('user'), indent=4)
            ex_assistant = json.dumps(example_prompt.get('assistant'), indent=4)
            # TODO: potentially also support multiple examples for batch prompts
            messages.append({"role": "user", "content": ex_user})
            messages.append({"role": "assistant", "content": ex_assistant})

    # Account for single file processing
    else:
        if use_cross_sample_information:
            batch_prompt = """You are a regular human known for your detailed motion descriptions and simple vocabulary. In the following, you will receive a task and several observations all describing the same human motion. Your task is it to reuse some of the information, depending on the task you receive, across all observations to refine the observation. Do not mention muscles.
            For each observation, generate a detailed version of the description individually by describing the full motion from start to finish each time. Refine each movement by mentioning each of the following body parts (even if they don't move): arms, legs, torso, neck, buttocks, and waist. Make sure that the movement descriptions do not contradict themsevels across observations.
            
            Format of input JSON that you will receive:
            {
                "filename": {
                    "observation1": description of the motion",
                    "observation2": description of the motion",
                    "observation3": description of the motion"
                }
            }

            Required output format:
            Your output must also be in JSON format and be in exactly the same data structure as the input JSON format. You should replace each observation individually. Each observation must be in a single string and must maintain the order of the motions as presented in the input. Each observation must be below 70 tokens/~60 words.
            """
            
        else:
            batch_prompt = """You are an average human known for your good motion descriptions and simple vocabulary.
            Keep your answers short in one, max two sentences. It must be below 70 tokens/~60 words.
            """
            
        if example_prompt:
            # Insert list of examples for user and assistant
            example_prompts = example_prompt.get('single')
            for example in example_prompts:
                ex_user = example.get('user')
                ex_assistant = example.get('assistent')
                messages.append({"role": "user", "content": ex_user})
                messages.append({"role": "assistant", "content": ex_assistant})
                
    new_system_prompt = batch_prompt + system_prompt
    # Prepend system prompt
    messages.insert(0, {"role": "system", "content": new_system_prompt})
    # Append user prompt at end
    messages.append({"role": "user", "content": data})

    new_prompt = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    refined_text = new_prompt.choices[0].message.content
    # print(data)
    # print(refined_text)
    # print("")
    
    if BATCH_PROCESSING or use_cross_sample_information:
        # Cut everything before the first '{' and after the last '}'
        refined_text = refined_text[refined_text.find('{'):refined_text.rfind('}')+1]
        # print(refined_text)
        return json.loads(refined_text)
    
    else:
        return refined_text


def refine_text(data_folder:str, 
                output_folder:str, 
                system_prompt:str, 
                example_prompt=None,
                batch_size:int=3, 
                model:str="gpt-3.5-turbo", 
                client=OpenAI(), 
                refine_specific_samples_txt_path=None,
                stop_after_n_batches=None,
                continue_previous=None,
                use_cross_sample_information=False
                ):
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

        if refine_specific_samples_txt_path is not None and stop_after_n_batches is not None:
            num_batches = min(num_batches, stop_after_n_batches)

        progress = tqdm(range(num_batches), desc="Processing batches")

        for i in progress:
            batch = files[i*batch_size:(i+1)*batch_size]

            try:
                input, annotations = process_data(batch)
                data = get_text_refinement(data=input, system_prompt=system_prompt, example_prompt=example_prompt, model=model, client=client, BATCH_PROCESSING=BATCH_PROCESSING)
                export_data(data=data, annotations_dict=annotations, output_folder=output_folder)

            except Exception as e:
                print(f"An error occurred while processing batch {i + 1} ({batch}): {e}")

            if stop_after_n_batches: # not None
                if i == stop_after_n_batches -1:
                    break

            progress.update(1)
        progress.close()

    elif use_cross_sample_information:
        num_files = len(files) if stop_after_n_batches is None else min(stop_after_n_batches, len(files))
        files = files[:num_files]

        for file in tqdm(files, desc=f"Processing files"):

            try:
                input, annotations = process_data([file], observation_instead_of_motion=True)
                data = get_text_refinement(data=input, system_prompt=system_prompt, example_prompt=example_prompt, model=model, client=client, BATCH_PROCESSING=BATCH_PROCESSING, use_cross_sample_information=True)
                export_data(data=data, annotations_dict=annotations, output_folder=output_folder)

            except Exception as e:
                    print(f"An error occurred while processing file {file}: {e}")
    # Process files one by one
    else:
        num_files = len(files) if stop_after_n_batches is None else min(stop_after_n_batches, len(files))
        files = files[:num_files]

        for file in tqdm(files, desc=f"Processing files"):

            try:
                input, annotations = process_data([file])

                # Convert input and annotions to dict
                input = json.loads(input)

                for motion, desc in input[file].items():

                    refined_text = get_text_refinement(data=desc, system_prompt=system_prompt, example_prompt=example_prompt, model=model, client=client, BATCH_PROCESSING=BATCH_PROCESSING)
                    input[file][motion] = refined_text
                    export_data(data=input, annotations_dict=annotations, output_folder=output_folder)

            except Exception as e:
                    print(f"An error occurred while processing file {file}: {e}")


    print("Text refinement complete.")

def main(args):

    if args.use_llama:
        print("Using Llama model. Make sure you specified the llama key")
        # Overwrite args with the ones from the config file

        # Set client and model
        base_url = 'http://localhost:11434/v1'
        api_key = args.api_key
        client = OpenAI(
            base_url = base_url,
            api_key=api_key
        )
    else:
        client = OpenAI()
    
    # set global variables to use throughout this file
    global DEVICE, HUMAN_ML_DIR, TEXTS_DIR
    DEVICE = args.DEVICE
    HUMAN_ML_DIR = args.HUMAN_ML_DIR
    TEXTS_DIR = args.texts_folder

    print(f"Using {DEVICE} device")

    # Set modelname to default if not specified
    model = getattr(args, 'model', 'gpt-3.5-turbo')
        
    # Load example prompt for model assistant and user
    if args.use_example:
        with open(os.path.join(args.ROOT_DIR, "prompts_examples" ,f"ex_{args.system_prompt}"), 'r') as file:
            example_prompt = json.load(file)
    else: example_prompt = None

    # Load system prompt
    if args.system_prompt[-5:] != ".json":
        args.system_prompt += ".json"
    system_prompt_path = os.path.join(args.ROOT_DIR, "prompts", args.system_prompt)
    with open(system_prompt_path, 'r') as file:
        system_prompt = json.load(file).get('system_prompt')

    # set refinement option to None if not specified
    if not args.refine_all_samples:
        refine_specific_samples_txt_path = args.source_file
    else:
        refine_specific_samples_txt_path = None

    refine_text(data_folder=args.texts_folder, 
            output_folder=args.target_folder,
            system_prompt=system_prompt,
            example_prompt=example_prompt,
            batch_size=args.batch_size,
            client=client,
            model=model,
            refine_specific_samples_txt_path=refine_specific_samples_txt_path,
            stop_after_n_batches=args.early_stopping,
            continue_previous=args.continue_previous,
            use_cross_sample_information=args.use_cross_sample_information,
        )
