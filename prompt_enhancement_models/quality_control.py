import os
import re

def write_failes_files(dataset_path:str, failed_files:list):

    output_path = os.path.join(dataset_path, "failed_files.txt")
    with open(output_path, 'w') as file:
        for failed_file in failed_files:
            file.write(f"{failed_file}\n")


def has_good_quality(file_path) -> bool:
    
    # print(f"Checking file {file_path}...")
    
    # Pattern checks end of line
    pattern = r".*#(\d+\.\d+)#(\d+\.\d+)$"

    # Read file
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            
            # Remove newline character
            line = line.strip()

            if not line: # Check if line is empty
                return False
            
            line = line.replace("#No annotation", "#0.0#0.0")
            # Check if line ends with correct pattern
            if not re.search(pattern, line):
                return False
            
    return True


def delete_failed_files(directory_path, failed_files):
    """Delete faulty files from dataset"""
    for filename in failed_files:
        file_path = os.path.join(directory_path, filename + ".txt")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File {filename} not found for deletion.")

    print(f"Dataset cleaned successfully (deleted {len(failed_files)}).")

    # return True

def check_dataset_quality(dataset_path, delete_faulty):

    failed_files = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(dataset_path, filename)
            if not has_good_quality(file_path):
                failed_files.append(filename.split(".")[0])

    write_failes_files(dataset_path, failed_files)

    if delete_faulty:
        delete_failed_files(dataset_path, failed_files)


delete_faulty = True
examples_folder = "prompt_enhancement_models/altered_texts_limb_specific_fixed"

check_dataset_quality(examples_folder, delete_faulty)