import spacy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
import torch
import clip
from typing import Tuple, Dict
from tqdm import tqdm
import pickle
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


def improved_prompt(text: str, nbrs: NearestNeighbors, index_to_text: Dict[int, str], model: object,
                    openai_client: OpenAI) -> str:
    """
    Given a motion description, finds similar motion descriptions in HumanML3D's training dataset,
    uses these as an additional input for a language model to create an enhanced prompt.
    TODO: current problems:
        - currently hardcoded for 2 neigbours
        - mirrored prompts (e.g. left only swapped with right) not detected

    @param openai_client:
    @param nbrs: NearestNeighbors object with embeddings
    @param index_to_text: dictionary mapping embedding index to original text
    @param text: motion description to refine
    @param model: embedding model to use, likely CLIP
    @return: enhanced motion description
    """
    # Get 2 nearest neighbors
    distances, indices = nbrs.kneighbors([text_to_embedding(text, model)])

    # print(f"Original: {text}\nExample 1: {index_to_text[indices[0][0]]}\nExample 2: {index_to_text[indices[0][1]]}")
    # print(distances[0][0], distances[0][1])

    # Filter based on distance (too close descriptions don't add value, too far will change semantics too much
    neighbors = [index_to_text[idx] for idx, dist in zip(indices[0], distances[0]) if 0.1 <= dist <= 0.4]
    # Remove duplicates
    neighbors = list(set(neighbors))

    if len(neighbors) == 2:
        print(f"Original Prompt: {text}\nNeighbor 1: {neighbors[0]}\nNeighbor 2: {neighbors[1]}")

    if len(neighbors) == 0:
        return text
    elif len(neighbors) == 1:
        prompt = (f"Adapt the following motion description: '{text}' "
                  f"so that it also includes elements of the motion description: "
                  f"'{neighbors[0]}'. "
                  f"Make sure the result is only one short sentence.")
    elif len(neighbors) == 2:
        prompt = (f"Adapt the following motion description: '{text}' "
                  f"so that it also includes elements of the motion descriptions "
                  f"'{neighbors[0]}' and '{neighbors[1]}'. "
                  f"Make sure the result is only one short sentence.")

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": prompt}
        ]
    )

    final_output = completion.choices[0].message.content.split("\n")[0]
    return final_output


def text_to_embedding(text: str, model: object) -> np.array:
    """
    Given a text, returns the feature vector by embedding with model

    @param text: Text to embed
    @param model: Embedding model to use, likely CLIP
    @return: normalized embedding vector consisting of 512 floats
    """
    # Encode text
    # Note that we are currently truncating long descriptions to fit the token length.
    # TODO: investigate how often this happens
    text = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()[0]


def dataset_to_neighbours(info_file_name: str, source_folder:str, model: object) -> Tuple[NearestNeighbors, Dict[int, str]]:
    """
    Given a text file specifying which files belong to the dataset split, return a fitted NearestNeighbours object
    containing the embeddings of all motion descriptions in these files along with a dictionary mapping
    embedding index to text.
    @param info_file_name: Text file specifying which files belong to the dataset split
    @param source_files: Folder in which all texts are saved to
    @param model: Embedding model to use, usually CLIP
    @return: NearestNeighbors object built with embeddings, dictionary mapping embedding index to text
    """
    embeddings = []
    index_to_text = {}
    num_lines = sum(1 for line in open(info_file_name))

    with open(info_file_name, 'r') as file:
        line_count = 0
        for line in tqdm(file, total=num_lines, desc="Generating embeddings"):
            file_name = line.strip()
            file_path = os.path.join(source_folder, file_name + ".txt")
            try:
                with open(file_path, 'r') as opened_file:
                    # Read the file line by line
                    for prompt_line in opened_file:
                        # Split the line by '#' to separate text and annotations
                        parts = prompt_line.strip().split('#')
                        # Extract the text part generate new prompt and part-of-speech tagging
                        text = parts[0].strip()
                        embeddings.append(text_to_embedding(text, model))
                        index_to_text[len(embeddings) - 1] = text
            except UnicodeDecodeError:
                # TODO: handle special characters in prompts, e.g. ", ° etc.
                print(f"Could not decode file {file_path}, skipping...")
            line_count += 1
            # if line_count >= 5:
            #     break

    embeddings_array = np.array(embeddings)
    knn_model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(embeddings_array)

    return knn_model, index_to_text


def text_enhancement(info_file_name: str, source_folder: str, target_folder: str, nbrs: NearestNeighbors, index_to_text: Dict[int, str], model: object,
                     openai_client: OpenAI, early_stopping=None) -> None:
    """
    Runs full text enhancement pipeline for all files specified by info_file_name and saves them at folder
    altered_texts.
    @param openai_client:
    @param info_file_name: Text file specifying which files belong to the dataset split
    @param source_folder: Folder in which all texts are
    @param target_folder: Folder to which enhanced texts are saved
    @param nbrs: NearestNeighbors object built with embeddings
    @param index_to_text: Dictionary mapping embedding index to text
    @param model: Embedding model to use, usually CLIP
    @early stopping: Stop enhancement after x steps for testing purposes
    """
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        print("Could not locate en_core_web_sm model, please install with \"python -m spacy download en_core_web_sm\"")
        exit(1)
    if early_stopping is not None:
        num_lines = min(sum(1 for line in open(info_file_name)), early_stopping)
    else:
        num_lines = sum(1 for line in open(info_file_name))
        
    with open(info_file_name, 'r') as file:
        line_count = 0
        for line in tqdm(file, total=num_lines, desc="Generating enhanced motion descriptions"):
            # Remove newline character and any leading/trailing whitespace
            file_name = line.strip()

            # Construct the full path to the file
            file_path = os.path.join(source_folder, file_name + ".txt")

            # Open the file
            altered_text_path = os.path.join(target_folder, file_name + ".txt")
            with open(file_path, 'r') as opened_file:
                # Clear the file
                open(altered_text_path, 'w').close()
                # Read the file line by line
                for prompt_line in opened_file:
                    # Split the line by '#' to separate text and annotations
                    parts = prompt_line.strip().split('#')
                    # Extract the text part generate new prompt and part-of-speech tagging
                    text = parts[0].strip()
                    tokenized_text = clip.tokenize(text, truncate=True).to(DEVICE)
                    text_features = model.encode_text(tokenized_text).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    new_prompt = improved_prompt(text, nbrs, index_to_text, model, openai_client)
                    word_list, pose_list = process_text(new_prompt, nlp)
                    new_prompt_tag = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
                    # Extract annotations
                    annotations = "#" + parts[2].strip().split('#')[0].strip() + "#" + parts[3].strip().split('#')[
                        0].strip()
                    with open(altered_text_path, 'a') as altered_file:
                        altered_file.write(new_prompt + '#' + new_prompt_tag + annotations + '\n')
            line_count += 1
            if line_count >= 10:
                break


def main(args):
    print(f"Using {DEVICE} device")

    # Load model
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    client = OpenAI()

    # Check if pickle file of embeddings exists, if yes unpickle and use, else compute and pickle
    pickle_file_path = "D:\\LLMs\\neighbors_index.pickle"
    if os.path.exists(pickle_file_path):
        try:
            with open(pickle_file_path, 'rb') as f:
                print(f"Found embeddings pickle file, using it instead of recomputing.")
                knn, index_to_text = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            knn, index_to_text = dataset_to_neighbours(args.source_file, args.texts_folder, model)
    else:
        print(f"Did not find embeddings pickle file, recomputing...")
        knn, index_to_text = dataset_to_neighbours(args.source_file, args.texts_folder, model)
        # Pickle the obtained data
        try:
            with open(pickle_file_path, 'wb') as f:
                pickle.dump((knn, index_to_text), f)
        except Exception as e:
            print(f"Error pickling data: {e}")

    text_enhancement(args.source_file, args.texts_folder, args.target_folder, knn, index_to_text, model, client, args.early_stopping)