import torch
import logging

from utils import DEVICE

class Evaluator():
    def __init__(self, data_loader, prompt_adaptor, t2m_model: torch.nn.Module):
        """Basic Evaluator class, can be used as parent class for more specific evaluators

        Args:
            data_loader (torch DataLoader): data_loader object containing evaluation data
            prompt_adaptor : prompt enhancement object with an "adapt_prompt" method that returns an adapted prompt as string
            t2m_model (torch.nn.Module): Text2Motion model that can generate motion from a string prompt input
        """
        self.data_loader = data_loader
        self.prompt_adaptor = prompt_adaptor
        self.t2m_model = t2m_model
        
        self.evaluation_metric = ... # TODO: Define the evaluation metric as a function

    def evaluate(self):
        t2m_model = self.t2m_model.to(DEVICE)
        losses = []
        # evaluate with text enhancement
        for input, ground_truth in self.data_loader.evaluation_data:
            prompt = self.prompt_adaptor.adapt_prompt(input)
            prompt.to(DEVICE)
            motion_out = t2m_model.generate_motion(prompt)
            loss = self.evaluation_metric(motion_out, ground_truth).to("cpu")
            losses.append(loss)
            
        # TODO: Average the losses
        loss_average = sum(losses) / len(losses)
        logging.log(logging.INFO, f"Average loss with {self.evaluation_metrics}: {loss_average}")