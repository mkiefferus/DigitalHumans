import os
import re
import shutil
import argparse
import codecs as cs
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")

def has_good_quality(file_path) -> bool:
    
    # Pattern checks end of line
    pattern = r".*#((\d+\.\d+)|nan)#((\d+\.\d+)|nan)$"

    # Read file content
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except:
        print(f"Failed to read file {file_path}.")
        return False

    # Replace "#No annotation" with "#0.0#0.0"
    modified_lines = [line.replace("#No annotation", "#0.0#0.0") for line in lines]

    # Write the modified content back to the file
    if lines != modified_lines: # Only write if there are changes
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

    # Check each line for the correct pattern
    for line in modified_lines:
        line = line.strip()

        if not line:  # Check if line is empty
            return False
        
        # Check if line ends with correct pattern
        if not re.search(pattern, line):
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

            # Account for potential faulty original file
            if not has_good_quality(destination_file_path):
                os.remove(destination_file_path)
                print(f"File {destination_file_path} is faulty and has been deleted.")

            else:
                replaced_counter += 1

        except FileNotFoundError:
            print(f"Original file {original_file_path} not found for replacement.")


    if replaced_counter < len(failed_files):
        print(f"Attended to clean dataset - {replaced_counter}/{len(failed_files)} files replaced.")
    elif replaced_counter == len(failed_files):
        print(f"Dataset cleaned successfully.")
    else:
        print("Failed to clean dataset - no files replaced.")

def replace_not_generated_test_files(adjusted_dataset_path):
    test_file_path = os.path.join(HUMAN_ML_DIR, "test.txt")
    original_dataset_path = os.path.join(HUMAN_ML_DIR, "texts")
    id_list = []
    with open(test_file_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
                id_list.append(line.strip())
    counter = 0
    for file in tqdm(id_list):
        adjusted_does_not_exist = os.path.exists(path = os.path.join(adjusted_dataset_path, file + ".txt"))
        original_exists = os.path.exists(os.path.join(original_dataset_path, file + ".txt"))
        if not adjusted_does_not_exist and original_exists:
            original_file_path = os.path.join(original_dataset_path, file + ".txt")
            shutil.copy(original_file_path, adjusted_dataset_path)
            counter +=1
    print(f"There were {counter} files missing for the test dataset, that were replaced by the original text files.")
    return

def check_same_amount_of_prompts(adjusted_dataset_path):
    """Compare files in folder_a and folder_b, adjust lines in folder_b files if necessary."""
    original_dataset_path = os.path.join(HUMAN_ML_DIR, "texts")
    files_a = [f for f in os.listdir(original_dataset_path) if os.path.isfile(os.path.join(original_dataset_path, f))]
    files_b = [f for f in os.listdir(adjusted_dataset_path) if os.path.isfile(os.path.join(adjusted_dataset_path, f))]

    common_files = set(files_a).intersection(set(files_b))

    counter = 0
    for file_name in common_files:
        file_a_path = os.path.join(original_dataset_path, file_name)
        file_b_path = os.path.join(adjusted_dataset_path, file_name)

        with open(file_a_path, 'r') as file_a, open(file_b_path, 'r') as file_b:
            lines_a = file_a.readlines()
            lines_b = file_b.readlines()

        len_a = len(lines_a)
        len_b = len(lines_b)
        
        while len_b < len_a:
            with open(file_b_path, 'w') as file_b:
                first_line_b = lines_b[0] if lines_b else ""
                lines_b.extend([first_line_b] * (len_a - len_b))
                file_b.writelines(lines_b)
            counter += 1
    print("amount of files changed that had uneven amount of prompts: ", counter)
    

def compare_flags_first_entry(adjusted_dataset_path):
    test_file_path = os.path.join(HUMAN_ML_DIR, "test.txt")
    original_dataset_path = os.path.join(HUMAN_ML_DIR, "texts")
    id_list = []
    with open(test_file_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
                id_list.append(line.strip())
    counter = 0
    for file in tqdm(id_list):
        try:
            file_original = open(os.path.join(original_dataset_path, file + '.txt')).readlines()
            file_altered = open(os.path.join(adjusted_dataset_path, file + '.txt')).readlines()
            idx = 0
            line = file_original[idx]
            line_split = line.strip().split('#')
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else float(f_tag)
            to_tag = 0.0 if np.isnan(to_tag) else float(to_tag)
            
            line_altered = file_altered[idx]
            line_split_altered = line_altered.strip().split('#')
            f_tag_altered = float(line_split_altered[2])
            to_tag_altered = float(line_split_altered[3])
            f_tag_altered = 0.0 if np.isnan(f_tag_altered) else float(f_tag_altered)
            to_tag_altered = 0.0 if np.isnan(to_tag_altered) else float(to_tag_altered)
            
            if f_tag_altered != f_tag or to_tag_altered != to_tag:
                line_split_altered[2] = str(f_tag)
                line_split_altered[3] = str(to_tag)
                file_altered[idx] = '#'.join(line_split_altered) + '\n'
                counter +=1
            with open(os.path.join(adjusted_dataset_path, file + '.txt'), "w") as f:
                f.writelines(file_altered)
        except IndexError as e:
            print("A file doesn't seem to have a sample in line 0.")
            print(f"Replacing file {os.path.join(adjusted_dataset_path, file + '.txt')} with the original.")
            # with open(os.path.join(adjusted_dataset_path, file + '.txt'), "w") as f:
            #     f.writelines(file)
    print("Number of flags replaced: ", counter)  


def save_faulty_names(dataset_path, failed_files):
    """Save names of faulty files to a text file"""
    faulty_names_out_path = os.path.join(dataset_path, "failed_files.txt")
    with open(faulty_names_out_path, "w") as file:
        for filename in failed_files:
            file.write(filename + "\n")

    return faulty_names_out_path


def check_dataset_quality(dataset_path, replace:bool, delete:bool, test:bool):

    print(f"Checking dataset quality...")

    failed_files = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt") and not filename.startswith("failed_files"):
            file_path = os.path.join(dataset_path, filename)
            if not has_good_quality(file_path):
                failed_files.append(filename.split(".")[0])

    # Ensure that replace has higher priority
    if replace:
        replace_failed_files(dataset_path, failed_files)
    elif delete:
        delete_failed_files(dataset_path, failed_files)
        
    else:
        print(f"Dataset quality check complete. {len(failed_files)} faulty files found.")
        
        if len(failed_files) != 0:
            # Save name of faulty files if no flagg given
            faulty_names_out_path = save_faulty_names(dataset_path, failed_files)
            print(f"No processing flag given. Dataset contains faulty files. List of faulty files saved at {faulty_names_out_path}.")
    if test:
        replace_not_generated_test_files(adjusted_dataset_path=dataset_path)
        # check_same_amount_of_prompts(adjusted_dataset_path=dataset_path)
        compare_flags_first_entry(adjusted_dataset_path=dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset quality and handle faulty files.")
    parser.add_argument("--data", type=str, help="Path to the generated dataset folder.")
    parser.add_argument("-r", action="store_true", help="Replace '#No annotation' with '#0.0#0.0'.")
    parser.add_argument("-d", action="store_true", help="Delete faulty files.")
    parser.add_argument("-t", action= "store_true", help="For the test set, copy original file if it doesn't exist and compare/adjust the flags of the first entry.")

    args = parser.parse_args()  

    check_dataset_quality(args.data, args.r, args.d, args.t)