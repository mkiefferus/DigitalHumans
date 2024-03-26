from gpt4all import GPT4All
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

# Download the necessary resources for part-of-speech tagging
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

# Define the path to the original prompt folder/ test text list/ the new prompt folder
folder_path = '/Users/axelwagner/Documents/T2M/HumanML3D/texts'
text_list = "/Users/axelwagner/Documents/T2M/HumanML3D/test.txt"
new_folder_path = "/Users/axelwagner/Documents/T2M/HumanML3D/altered_texts"
#llm settings
device = "cpu" # "gpu"
llm_model = "mistral-7b-openorca.gguf2.Q4_0.gguf"
llm_save_path = None

if llm_save_path is not None:
    if not os.path.exists(os.path.dirname(llm_save_path)):
        os.mkdir(llm_save_path)
        print(f"Folder '{llm_save_path}' created successfully.")

#download llm model
model = GPT4All(llm_model,allow_download=True,model_path=llm_save_path,device=device)
#prompt settings
system_prompt = '### System:You are an AI assistant that translate the motion described by the given sentences to the motion of each bodypart only using one cohesive paragraph. The available body parts include [‘arms’, ‘legs’, ‘torso’, ‘neck’, ‘buttocks’, ‘waist’]..'
# system_prompt = '### System:You are an AI assistant and give more detailed descriptions of the motion described in the given text.' # prompt for a more detailed desciption but without explicit limb motion description
# prompt_template = '### User:{0}### Response:'
prompt_template = ''

#def improved_prompt --> improve text prompt
def improved_prompt(text,llm_model,device,system_prompt,prompt_template):
    output = None
    model = GPT4All(llm_model,allow_download=False,device=device)
    with model.chat_session(system_prompt=system_prompt, prompt_template=prompt_template):
        response1 = model.generate(prompt=text, temp=0,max_tokens=1000)
        output = model.current_chat_session[-1]
    final_output = output["content"].split("\n")[0]
    return final_output

def convert_to_string(word_pos_list):
    result = ' '
    for word, pos in word_pos_list:
        result += word + '/' + pos + ' '
    return '#' + result.strip()

#def tagset --> part-of-speech tagging
def tagset(text):
    tagged_tokens = pos_tag(word_tokenize(text), tagset='universal') 
    tagged_text = convert_to_string(tagged_tokens)

    return tagged_text

# Create a new folder
if not os.path.exists(os.path.dirname(new_folder_path)):
    os.mkdir(new_folder_path)
    print(f"Folder '{new_folder_path}' created successfully.")

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
 
        line_count += 1
        if line_count >= 4:
            break   