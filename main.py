# Main file to run the evaluation of the different models and prompt adaptation techniques

import argparse
import itertools
import re

from utils import SESSION_ID, init_logging
from pipeline_evaluator.evaluator_standard import Evaluator
from prompt_enhancement import Mistral

def parse_args():
    # all args that cannot be matched to the Trainer or DataLoader classes and are

    meta_args = ['experiment_name', 'e', 'run_name', 'r', 'train', 't']
    dataloader_args = ['dataset', 'd', 'split', 's'] # args for the dataloader (split for later end2end training)
    t2m_args = ['model', 'm'] # args for the text2motion model
    prompt_args = ['prompt_adaptation', 'p'] # args for the prompt adaptation

    parser = argparse.ArgumentParser(description='Implementation of ETHZ Digital Humans Project for improving Text to Motion generations by enhancing the text prompt.')
    parser.add_argument('-e', '--experiment_name', type=str, required=True, 
                        help="The experiment name. Default to the model name and prompt adaptation technique \
                        with an additional integer value of the current time.")
    parser.add_argument('-r', '--run_name', type=str, required=True, 
                        help="The run name, which can be used for logging, in order to distinguish runs of the same \
                        experiment.")
    parser.add_argument('-t', '--train', type=bool, default=False,
                        help="Set to true if you want to train the model end-to-end") #TODO
    parser.add_argument('-d', '--dataset', type=str, required=False, default='human_ml_3d',
                        help="The name of the dataset. If not available locally, the dataset will be downloaded.")
    parser.add_argument('-s', '--split', type=float, required=False, default=0.0, 
                        help="Only relevant if 'train' is set to True. The percentile of the training data used for actual training. \
                        The rest will be used for evaluation.")
    parser.add_argument('-m', '--model', type=str, required=False, default='mo_mask',
                        help="The name of the Text-to-Motion model to be used.")
    parser.add_argument('-p', '--prompt_adaptation', type=str, required=False, default="mistral",
                        help="The name of the prompt adaptation technique to be used.")
    known_args, unknown_args = parser.parse_known_args()
    
    # Process the commandline arguments
    remove_leading_dashes = lambda s: ''.join(itertools.dropwhile(lambda c: c == '-', s))
    # float check taken from https://thispointer.com/check-if-a-string-is-a-number-or-float-in-python/
    cast_arg = lambda s: s[1:-1] if s.startswith('"') and s.endswith('"') \
        else int(s) if remove_leading_dashes(s).isdigit() \
        else float(s) if re.search('[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', s) is not None \
        else s.lower() == 'true' if s.lower() in ['true', 'false'] \
        else None if s.lower() == 'none' \
        else eval(s) if any([s.startswith('(') and s.endswith(')'),
                             s.startswith('[') and s.endswith(']'),
                             s.startswith('{') and s.endswith('}')]) \
        else s
    
    # check if experiment name is set, otherwise create automatically
    if known_args.experiment_name is None:
        known_args.experiment_name = f"{known_args.prompt_adaptation}_{known_args.model}_{SESSION_ID}"
    
    # 
    known_args_dict = dict(map(lambda arg: (arg, getattr(known_args, arg)), vars(known_args)))
    
    # specify the arguments for the experiment, dataloader, t2m-model and prompt adaptation
    meta_args = {k: v for k, v in known_args_dict.items() if k.lower() in meta_args}
    dataloader_args = {k: v for k, v in known_args_dict.items() if k.lower() in dataloader_args}
    t2m_args = {k: v for k, v in known_args_dict.items() if k.lower() in t2m_args}
    prompt_args = {k: v for k, v in known_args_dict.items() if k.lower() in prompt_args}
    
    return meta_args, dataloader_args, t2m_args, prompt_args
    
if __name__ == '__main__':
    meta_args, dataloader_args, t2m_args, prompt_args = parse_args()
    
    # init log file
    init_logging(meta_args['experiment_name'], meta_args['run_name'])
    
    # Note: If this gets too complicated, consider using a factory pattern
    # if dataloader_args["dataset"] == "human_ml_3d":
        # dataloader = ...
    
    if prompt_args["prompt_adaptation"] == "mistral":
        prompt_enhancer = Mistral()
        
    # if t2m_args["model"] == "mo_mask":
        # t2m = ...
    
    # evaluator = Evaluator(dataloader, prompt_enhancer, t2m)
    # evaluator.evaluate()