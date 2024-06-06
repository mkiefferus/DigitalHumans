import os
import shutil
import argparse
from tqdm import tqdm
from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")

def semantic_check(file_path:str, client, model:str) -> bool:
    """Compare original prompt with refined prompt and check whether they are semantically similar enough."""
    # Print file name (without full path)
    _, file_name = os.path.split(file_path)

    # Open refined file
    try:
        with open(file_path, 'r') as file:
            lines_refined = file.readlines()
    except:
        print(f"Failed to read refined file {file_path}.")
        return False
    
    # Open corresponding original file
    original_file_path = os.path.join(HUMAN_ML_DIR, "texts", file_name)
    try:
        with open(original_file_path, 'r') as file:
            lines_original = file.readlines()
    except:
        print(f"Failed to read original file {file_name}, skipping...")
        return True

    # Check if the number of lines is the same
    if len(lines_refined) != len(lines_original):
        print(f"Number of lines in refined and original files do not match, run quality_control first, skipping...")
        return True

    system_prompt = """You will be given two motion descriptions, an original and a refined one.
        Please compare the two and answer with "Yes" if the refined motion description is semantically roughly equivalent to the original motion description, otherwise answer with "No".
        It is fine if the structure of the sentences is different and details are added or removed, as long as the most important meaning is preserved.
        Focus only on very important aspects of the motion and always include "Yes" or ("No" + reason) in your answer, even if you compared the same motions previously!
        In case you answer with "No", add the reason in brackets afterwards."""
    
    examples = [
        (
            ("Original: a person performs a typical broadjump\n"
            "Refined: The person bends their arms and crouches down preparing for a jump, then extend their arms back as they propel themselves forward with their legs."),
            "Yes"
        ),
        (
            ("Original: a person kicks with their right leg\n"
            "Refined: The person's right leg extends forward swiftly for a kick."),
            "Yes"
        ),
        (
            ("Original: a person slowly walks in a counter clockwise circle\n"
            "Refined: The person's arms are relaxed and swing gently with each step."),
            "No (missing circle motion information)"
        )
    ]

    for i in range(len(lines_refined)):
        refined_line = lines_refined[i].split("#")[0].strip()
        original_line = lines_original[i].split("#")[0].strip()

        messages = []
        # System Prompt
        messages.append({"role": "system", "content": system_prompt})
        # Examples
        for ex_user, ex_assistant in examples:
            messages.append({"role": "user", "content": ex_user})
            messages.append({"role": "assistant", "content": ex_assistant})
        # User Prompt
        messages.append({"role": "user", "content": f"Original: {original_line}\nRefined: {refined_line}"})

        prompt = client.chat.completions.create(
            model=model,
            messages=messages
        )

        # Check assistant answer
        answer = prompt.choices[0].message.content

        if "No" in answer:
            print(file_name)
            # Print both lines and result
            print(f"Original: {original_line}")
            print(f"Refined: {refined_line}")
            print(f"Assistant: {answer}")
            print("\n")
            return False

    return True


def delete_failed_files(dataset_path, failed_files):
    """Delete faulty files from dataset"""
    for filename in failed_files:
        file_path = os.path.join(dataset_path, filename + ".txt")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File {filename} not found for deletion.")

    deleted_files_list = save_faulty_names(dataset_path, failed_files)
    print(f"Dataset cleaned successfully (deleted {len(failed_files)}). List of deleted files saved at {deleted_files_list}.")


def replace_failed_files(dataset_path, failed_files):
    """Replace faulty files with originals from the org_data_folder"""

    replaced_counter = 0
    for filename in failed_files:
        original_file_path = os.path.join(HUMAN_ML_DIR, "texts", filename + ".txt")
        destination_file_path = os.path.join(dataset_path, filename + ".txt")

        try:
            shutil.copy(original_file_path, destination_file_path)
        except FileNotFoundError:
            print(f"Original file {original_file_path} not found for replacement.")


    if replaced_counter < len(failed_files):
        print(f"Attended to clean dataset - {replaced_counter}/{len(failed_files)} files replaced.")
    elif replaced_counter == len(failed_files):
        print(f"Dataset cleaned successfully.")
    else:
        print("Failed to clean dataset - no files replaced.")


def save_faulty_names(dataset_path, failed_files):
    """Save names of faulty files to a text file"""
    faulty_names_out_path = os.path.join(dataset_path, "failed_files.txt")
    with open(faulty_names_out_path, "w") as file:
        for filename in failed_files:
            file.write(filename + "\n")

    return faulty_names_out_path


def check_dataset_semantics(dataset_path, replace:bool, delete:bool, client, model:str):

    print(f"Checking dataset quality...")
    failed_files = []

    for filename in tqdm(os.listdir(dataset_path)):
        if filename.endswith(".txt") and not filename.startswith("failed_files"):
            file_path = os.path.join(dataset_path, filename)
            if not semantic_check(file_path, client, model):
                failed_files.append(filename.split(".")[0])

    # Print fraction of faulty files
    print(f"Dataset quality check complete. {len(failed_files)}/{len(os.listdir(dataset_path))} files faulty.")

    # Ensure that replace has higher priority
    if replace:
        replace_failed_files(dataset_path, failed_files)
    elif delete:
        delete_failed_files(dataset_path, failed_files)
    else:       
        if len(failed_files) != 0:
            # Save name of faulty files if no flag given
            faulty_names_out_path = save_faulty_names(dataset_path, failed_files)
            print(f"No processing flag given. Dataset contains faulty files. List of faulty files saved at {faulty_names_out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset quality and handle faulty files.")
    parser.add_argument("--data", type=str, help="Path to the generated dataset folder.", required=True)
    parser.add_argument("--model", type=str, help="LLM to use for checking quality, either llama3 or gpt-3.5-turbo", default="llama3")
    parser.add_argument("-r", action="store_true", help="Replace faulty files with original files.")
    parser.add_argument("-d", action="store_true", help="Delete faulty files.")

    args = parser.parse_args()  

    client = OpenAI()
    if (args.model == "llama3"):
        print("Using llama3 model for checking similarity")
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
        )
    elif (args.model == "gpt-3.5-turbo"):
        print("Using GPT-3.5 Turbo model for checking similarity")


    check_dataset_semantics(args.data, args.r, args.d, client, args.model)