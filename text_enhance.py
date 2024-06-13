# Main file to run different prompt enhancement techniques

import argparse
import subprocess
import logging
import os

from utils import SESSION_ID, MOMASK_REPO_DIR, HUMAN_ML_DIR, DEVICE, ROOT_DIR
from utils.logging import init_logging, end_logging
from prompt_enhancement.similarity_search_openai import main as similarity_search_openai
from prompt_enhancement.text_refinement import main as regular_text_refinement
from prompt_enhancement.quality_control import check_dataset_quality

def parse_args():
    # all args that cannot be matched to the Trainer or DataLoader classes and are
    parser = argparse.ArgumentParser(description='Implementation of ETHZ Digital Humans Project for enhancing the text prompt.')
    parser.add_argument('-e', '--experiment_name', type=str, required=False, default=None,
                        help="The experiment name. Default to the prompt adaptation technique \
                        with an additional value of the current time.")
    parser.add_argument('-pa', '--prompt_adaptation', type=str, required=False, default="regular",
                        help="The name of the prompt adaptation technique to be used. Available: similarity, regular")
    parser.add_argument('-qc', '--quality_control_only', action="store_true", required=False,
                        help="Whether to only perform quality control on the files in the dataset.")
    parser.add_argument('-sp', '--system_prompt', type=str, required=False, default="added_details",
                        help="The name of the system prompt to be given to the prompt adaptation model, as can be viewed in folder 'prompts'.")
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        help="Whether to output information into the console (True) or the logfile (False).")
    parser.add_argument('-s', '--early_stopping', type=int, required=False, default=None,
                        help="Whether to stop the refinement after x steps for testing purposes.")
    
    # below args are only necessary for regular enhancement, not similarity search
    parser.add_argument("--batch_size", type=int, default=1, help="If larger than 1, the model will process multiple files at once.")
    parser.add_argument("--continue_previous", type=str, default=None, help="Continue refining texts from a specific folder")
    parser.add_argument("--refine_all_samples", action="store_true", required=False, help="Refine all samples. Default: refine test samples only")
    parser.add_argument("--samples_text_file", type=str, default="test.txt", help="Text file specifying samples to refine. Default: test.txt")
    parser.add_argument("--use_cross_sample_information",  action="store_true", required=False, help="Use information from multiple samples of the same text file to output enhanced samples with more information. Makes batch_size arg invalid")
    parser.add_argument("--use_example", action="store_true", required=False, help="Whether to use example prompts for the model assistant and user (specified as ex_<system_prompt>.json) in folder prompts_examples")
    parser.add_argument("--use_llama", action="store_true", required=False, help="Use Llama model")
    parser.add_argument("--llama_key", type=str, default="ollama", help="Key for the llama model")
    
    # below args are for quality control
    parser.add_argument("-r", "--replace", action="store_true", help="Replace refined files with original ones if fail quality check.")
    parser.add_argument("-d", "--delete", action="store_true", help="Delete faulty files.")
    parser.add_argument("-t", action="store_true", help="For the test set, copy original file if it doesn't exist and compare/adjust the flags of the first entry.")
    args = parser.parse_args()
    
    known_args, _ = parser.parse_known_args()
    
    # add relevant paths to arg parser, as scripts do not have access to the utils.py file
    known_args.SESSION_ID = SESSION_ID
    known_args.ROOT_DIR = ROOT_DIR
    known_args.MOMASK_REPO_DIR = MOMASK_REPO_DIR
    known_args.HUMAN_ML_DIR = HUMAN_ML_DIR
    known_args.DEVICE = DEVICE
    
    # check if experiment name is set, otherwise create automatically
    if known_args.experiment_name is None:
        known_args.experiment_name = f"{known_args.prompt_adaptation}_{known_args.system_prompt}_{SESSION_ID}"
    
    # specify the arguments for the experiment, dataloader, t2m-model and prompt adaptation
    
    return known_args
    
if __name__ == '__main__':
    args = parse_args()
    
    # init log file
    logfile = init_logging(args.experiment_name, args.verbose)
    
    # Source and output folder structure
    args.target_folder = os.path.join(HUMAN_ML_DIR, f"texts_{args.prompt_adaptation}_{args.system_prompt}_{SESSION_ID}")
    args.target_folder = args.continue_previous if args.continue_previous is not None else args.target_folder
    os.makedirs(args.target_folder, exist_ok=True)
    args.source_file = os.path.join(HUMAN_ML_DIR, args.samples_text_file)
    args.texts_folder = os.path.join(HUMAN_ML_DIR, "texts")
    
    # Warn if no quality control flag is given
    if not args.replace and not args.delete:
        raise Warning("No cleanup/quality control flag given. Use -r to replace or -d to delete possible faulty files.")
    
    if args.quality_control_only:
        # perform quality control on the files in the dataset
        if args.continue_previous:
            check_dataset_quality(args, test=False)
        else:
            raise Exception("Quality control only possible with a previous run. Specify the file path with the generated texts with --continue_previous")
    
    else:
        if args.prompt_adaptation == "similarity":
            # perform similarity search
            similarity_search_openai(args)
        
        elif args.prompt_adaptation == "regular":
            regular_text_refinement(args)
            pass
        
        if args.replace or args.delete:
            # Check dataset quality
            check_dataset_quality(args,test=False) # TODO: adjust test flag
    
    # finish logging
    logging.info("Finished with the following args:")
    logging.info(args)
    end_logging(logfile)