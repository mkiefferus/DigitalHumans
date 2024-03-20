from gpt4all import GPT4All
from utils import PROMPT_MODEL_FILES_DIR

class Mistral():
    def __init__(self) -> None:
        self.model = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf", model_path=PROMPT_MODEL_FILES_DIR)
        self.prompt_task = "In your answer, only directly answer the question in one paragraph. Translate the following motion by the given sentences to the motion of each bodypart. The available body partsinclude [‘arms’, ‘legs’, ‘torso’, ‘neck’, ‘buttocks’, ‘waist’]."
    
    def adapt_prompt(self, input: str) -> str:
        with self.model.chat_session():
            response = self.model.generate(f"{self.prompt_task} {input}", temp=0, max_tokens=1000)
        return response