import os
import shutil
import argparse
from tqdm import tqdm
from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")

def semantic_check(file_path:str, client, model:str, replace:bool, verbose:bool):
    """Compare original prompts in file with refined prompts and check whether they are semantically similar enough.
       Return (whether file contains errors, number of failed prompts) in file.
    """
    # Print file name (without full path)
    _, file_name = os.path.split(file_path)

    # Open refined file
    with open(file_path, 'r') as file:
        lines_refined = file.readlines()
    
    # Open corresponding original file
    original_file_path = os.path.join(HUMAN_ML_DIR, "texts", file_name)
    with open(original_file_path, 'r') as file:
        lines_original = file.readlines()

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

    invalid_refinements = 0

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
            if (verbose):
                print(file_name)
                # Print both lines and result
                print(f"Original: {original_line}")
                print(f"Refined: {refined_line}")
                print(f"Assistant: {answer}")
                print("\n")

            invalid_refinements += 1
            if (replace):
                # Replace refined line with original line
                lines_refined[i] = lines_original[i]
    
    # Save refined file with replaced lines
    if (replace):
        with open(file_path, 'w') as file:
            file.writelines(lines_refined)

    return (len(lines_refined), invalid_refinements)


def save_faulty_names(dataset_path, failed_files):
    """Save names of faulty files to a text file"""
    faulty_names_out_path = os.path.join(dataset_path, "failed_files.txt")
    with open(faulty_names_out_path, "w") as file:
        for filename in failed_files:
            file.write(filename + "\n")

    return faulty_names_out_path


def check_dataset_semantics(dataset_path, replace:bool, client, model:str, verbose:bool):
    print(f"Checking dataset quality...")
    failed_files = []
    prompt_count = 0
    invalid_refinement_count = 0

    for filename in tqdm(os.listdir(dataset_path)):
        if filename.endswith(".txt") and not filename.startswith("failed_files"):
            file_path = os.path.join(dataset_path, filename)
            try:
                prompt_amount, invalid_refinement_amount = semantic_check(file_path, client, model, replace, verbose)
                prompt_count += prompt_amount
                invalid_refinement_count += invalid_refinement_amount
                if (invalid_refinement_amount > 0):
                    failed_files.append(filename.split(".")[0])
            except:
                print(f"Failed to check file {filename}.")

    # Print fraction of faulty files
    print(f"Dataset quality check complete. {invalid_refinement_count}/{prompt_count} refined prompts invalid.")

    # Save name of files containing invalid refinements
    if len(failed_files) != 0:
        faulty_names_out_path = save_faulty_names(dataset_path, failed_files)
        print(f"List of files containing invalid refinements saved at {faulty_names_out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset quality and handle faulty files.")
    parser.add_argument("--data", type=str, help="Path to the generated dataset folder.", required=True)
    parser.add_argument("--model", type=str, help="LLM to use for checking quality, either llama3 or gpt-3.5-turbo", default="llama3")
    parser.add_argument("-r", action="store_true", help="Replace faulty prompt refinements with original texts.")
    parser.add_argument("-v", action="store_true", help="Verbose mode.")

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
    else:
        raise Warning("Invalid model name, choose either llama3 or gpt-3.5-turbo")


    check_dataset_semantics(args.data, args.r, client, args.model, args.v)