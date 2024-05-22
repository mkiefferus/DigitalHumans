import os
import re
import shutil
import argparse

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


def save_faulty_names(dataset_path, failed_files):
    """Save names of faulty files to a text file"""
    faulty_names_out_path = os.path.join(dataset_path, "failed_files.txt")
    with open(faulty_names_out_path, "w") as file:
        for filename in failed_files:
            file.write(filename + "\n")

    return faulty_names_out_path


def check_dataset_quality(dataset_path, replace:bool, delete:bool):

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
            print(f"No processing flagg given. Dataset contains faulty files. List of faulty files saved at {faulty_names_out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset quality and handle faulty files.")
    parser.add_argument("--data", type=str, help="Path to the dataset folder.")
    parser.add_argument("-r", action="store_true", help="Replace '#No annotation' with '#0.0#0.0'.")
    parser.add_argument("-d", action="store_true", help="Delete faulty files.")

    args = parser.parse_args()  

    check_dataset_quality(args.data, args.r, args.d)