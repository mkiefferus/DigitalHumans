from gpt4all import GPT4All
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm
from utils import PROMPT_MODEL_FILES_DIR, DEVICE, OUT_DIR

# Download the necessary resources for part-of-speech tagging
nltk.download('punkt', download_dir=PROMPT_MODEL_FILES_DIR, quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir=PROMPT_MODEL_FILES_DIR, quiet=True)
nltk.download('universal_tagset', download_dir=PROMPT_MODEL_FILES_DIR, quiet=True)

def convert_to_string(word_pos_list) -> str:
    """Converts tagset to string (helperfunction)"""
    result = ' '
    for word, pos in word_pos_list:
        result += word + '/' + pos + ' '
    return '#' + result.strip()

#def tagset --> part-of-speech tagging
def tagset(text):
    """Generates tags for text prompt."""
    tagged_tokens = pos_tag(word_tokenize(text), tagset='universal') 
    tagged_text = convert_to_string(tagged_tokens)

    return tagged_text

def main():
    # M = Mistral()
    data_path = "MoMask/dataset/HumanML3D/texts_org"
    test_data_filter_path = "MoMask/dataset/HumanML3D/test.txt"
    output_path = os.path.join(OUT_DIR, f"text_refinements/Mistral/joints")

    # num_files = sum(1 for line in open(M.test_data_filter_path).close())
    num_files = sum(1 for line in open(test_data_filter_path))
    
    # Iterate through all test files
    with open(test_data_filter_path, 'r') as file:


        line_count = 0
        for file in tqdm(file, total=num_files, desc="Processing files"):
            # Remove newline character and any leading/trailing whitespace
            file_name = file.strip()
            
            # Construct the full path to the file
            file_path = os.path.join(data_path, file_name+".txt")
            
            # Check if the file exists
            if os.path.exists(file_path):

                # Open the file
                file_output_path = os.path.join(output_path, file_name+".txt")

                with open(file_path, 'r') as current_file: 

                    # Clear the file
                    open(file_output_path, 'w').close()

                    # Read the file line by line
                    for prompt_line in current_file: 

                        # Split the line by '#' to separate text and annotations
                        parts = prompt_line.strip().split('#')
                        
                        # Extract the text part generate new prompt and part-of-speech tagging
                        text = parts[0].strip()
                        new_prompt = M.adapt_prompt(text)
                        new_prompt_tag = tagset(new_prompt)

                        # Extract annotations
                        first_annotation = parts[2].strip().split('#')[0].strip()
                        second_annotation = parts[3].strip().split('#')[0].strip()
                        annotations = "#" + first_annotation + "#" + second_annotation

                        with open(file_output_path, 'a') as altered_file:
                            altered_file.write(new_prompt + new_prompt_tag + annotations + '\n')

            line_count += 1
            if line_count >= 4:
                break   

if __name__ == "__main__":
    main() 