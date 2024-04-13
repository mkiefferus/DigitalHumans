from gpt4all import GPT4All
from utils.utils import PROMPT_MODEL_FILES_DIR, DEVICE, OUT_DIR
import os
from tqdm import tqdm

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


class Mistral:
    def __init__(self) -> None:
        self.dir_path = PROMPT_MODEL_FILES_DIR

        # Check if model folder is given
        if self.dir_path is not None:
            if not os.path.exists(os.path.dirname(self.dir_path)):
                os.mkdir(self.dir_path)
                print(f"Folder '{self.dir_path}' created successfully.")

        self.model = GPT4All(
            "mistral-7b-openorca.gguf2.Q4_0.gguf",
            allow_download=True,
            model_path=self.dir_path,
            device=DEVICE,
            )

        # Prompts
        self.system_prompt = "### System:You are an AI assistant that translate the motion described by the given sentences to the motion of each bodypart only using one cohesive paragraph. The available body parts include ['arms', 'legs', 'torso', 'neck', 'buttocks', 'waist']."
        # self.system_prompt = "### System:You are an AI assistant and give more detailed descriptions of the motion described in the given text." # prompt for a more detailed desciption but without explicit limb motion description

        self.prompt_template = ""

        # Data
        self.data_path = "MoMask/dataset/HumanML3D/texts_org"
        self.test_data_filter_path = "MoMask/dataset/HumanML3D/test.txt"
        self.output_path = os.path.join(OUT_DIR, f"{self.__class__.__name__}/joints")

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            print(f"Output folder '{self.output_path}' created successfully.")



    def adapt_prompt(self, input: str) -> str:
        """Refines simple text prompt"""

        with self.model.chat_session(
            system_prompt=self.system_prompt, 
            prompt_template=self.prompt_template
            ):

            response = self.model.generate(prompt=input, temp=0, max_tokens=1000)
            output = self.model.current_chat_session[-1]
        
        final_output = output['content'].split("\n")[0]
        return final_output