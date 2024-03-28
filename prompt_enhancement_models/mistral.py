from gpt4all import GPT4All
from utils import PROMPT_MODEL_FILES_DIR

class PromptRefiner():
    def __init__(self, input_texts, output_texts, system_prompt) -> None:
        self.input_texts_paths = input_texts
        self.input_texts_paths = output_texts
        # TODO: load system prompt from json
        
        self.system_prompt = system_prompt
        

class Mistral(PromptRefiner):
    def __init__(self, input_texts, output_texts, system_prompt) -> None:
        super().__init__(input_texts, output_texts, system_prompt)
        self.model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf", model_path=PROMPT_MODEL_FILES_DIR)
    
    def adapt_prompt(self, input: str) -> str:
        with self.model.chat_session():
            response = self.model.generate(f"{self.prompt_task} {input}", temp=0, max_tokens=1000)
        return response