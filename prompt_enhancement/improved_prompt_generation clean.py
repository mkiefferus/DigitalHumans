from gpt4all import GPT4All
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm
from utils import PROMPT_MODEL_FILES_DIR

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
    num_lines = sum(1 for line in open(text_list))
    # Read the input text file line by line
    with open(text_list, 'r') as file:
        line_count = 0
        for line in tqdm(file,total=num_lines, desc="Processing files"):
            # Remove newline character and any leading/trailing whitespace
            file_name = line.strip()
            
            # Construct the full path to the file
            file_path = os.path.join(folder_path, file_name+".txt")
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Open the file
                altered_text_path = os.path.join(new_folder_path, file_name+".txt")
                with open(file_path, 'r') as opened_file: 
                    # Clear the file
                    open(altered_text_path, 'w').close()
                    # Read the file line by line
                    for prompt_line in opened_file: 
                        # Split the line by '#' to separate text and annotations
                        parts = prompt_line.strip().split('#')
                        # Extract the text part generate new prompt and part-of-speech tagging
                        text = parts[0].strip()
                        new_prompt = improved_prompt(text,llm_model,device,system_prompt,prompt_template)
                        new_prompt_tag = tagset(new_prompt)
                        # Extract annotations
                        annotations = "#"+parts[2].strip().split('#')[0].strip()+"#"+parts[3].strip().split('#')[0].strip()
                        with open(altered_text_path, 'a') as altered_file:
                            altered_file.write(new_prompt+new_prompt_tag+annotations + '\n')



if __name__ == "__main__":
    main() 