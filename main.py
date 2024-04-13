# Main file to run the evaluation of the different models and prompt adaptation techniques

import argparse
import subprocess
import logging
import os

from utils.utils import SESSION_ID, MOMASK_REPO_DIR, OUT_DIR
from utils.logging import init_logging, end_logging
from prompt_enhancement_models import Mistral

def parse_args():
    # all args that cannot be matched to the Trainer or DataLoader classes and are
    parser = argparse.ArgumentParser(description='Implementation of ETHZ Digital Humans Project for improving Text to Motion generations by enhancing the text prompt.')
    parser.add_argument('-e', '--experiment_name', type=str, required=True, 
                        help="The experiment name. Default to the model name and prompt adaptation technique \
                        with an additional integer value of the current time.")
    parser.add_argument('-r', '--run_name', type=str, required=True, 
                        help="The run name, which can be used for logging, in order to distinguish runs of the same \
                        experiment.")
    parser.add_argument('-t', '--train', type=bool, default=False,
                        help="Set to true if you want to train the model end-to-end") #TODO
    parser.add_argument('-d', '--dataset', type=str, required=False, default='humanml3d',
                        help="The name of the dataset. If not available locally, the dataset will be downloaded.")
    parser.add_argument('-s', '--split', type=float, required=False, default=0.0, 
                        help="Only relevant if 'train' is set to True. The percentile of the training data used for actual training. \
                        The rest will be used for evaluation.")
    parser.add_argument('-m', '--model', type=str, required=False, default='mo_mask',
                        help="The name of the Text-to-Motion model to be used.")
    parser.add_argument('-pa', '--prompt_adaptation_model', type=str, required=False, default=None,
                        help="The name of the prompt adaptation technique to be used.")
    parser.add_argument('-sp', '--system_prompt', type=str, required=False, default="optimal",
                        help="The name of the system prompt to be given to the prompt adaptation model, as can be viewed in folder 'prompts'.")
    
    known_args, _ = parser.parse_known_args()
    
    # check if experiment name is set, otherwise create automatically
    if known_args.experiment_name is None:
        known_args.experiment_name = f"{known_args.prompt_adaptation_model}_{known_args.model}_{SESSION_ID}"
    
    # specify the arguments for the experiment, dataloader, t2m-model and prompt adaptation
    
    return known_args
    
if __name__ == '__main__':
    args = parse_args()
    
    # init log file
    logfile = init_logging(args.experiment_name, args.run_name)
    print(logfile)
    
    args.out_dir = os.path.join(OUT_DIR, args.experiment_name)
    # os.makedirs(args.out_dir, exist_ok=True)
    
    # Note: If this gets too complicated, consider using a factory pattern
    if not args.train:
        # prompt adaptation
        if args.prompt_adaptation_model:
            if args.dataset == "humanml3d" and args.model == "mo_mask":
                args.dataset_path = os.path.join(MOMASK_REPO_DIR, "data", "t2m")
            elif args.dataset == "kit" and args.model == "mo_mask":
                args.dataset_path = os.path.join(MOMASK_REPO_DIR, "data", "kit")
            args.texts_path = os.path.join(args.dataset_path, "texts")
            
            # precompute the promt adaptation
            args.adapted_prompts_path = os.path.join(args.dataset_path, f"texts_{args.prompt_adaptation_model}_{args.system_prompt}")
            if not os.path.exists(args.adapted_prompts_path) or len(os.listdir(args.adapted_prompts_path)) == 0: # no adapted prompts available yet
                # create folder if not exists
                os.makedirs(args.adapted_prompts_path, exist_ok=True)
                # Perform prompt adaptations and save in created folder
                if args.prompt_adaptation_model == "mistral":
                    pass
                    #prompt_enhancer = Mistral(input_texts = args.texts_path, output_texts = args.adapted_prompts_path, system_prompt = args.system_prompt)
                    #prompt_enhancer.adapt_prompt() # TODO @Max,Axel
            
            # Rename the data folders so that MoMask and other T2M models automatically use the adapted prompt
            altered_original_folder = os.path.join(args.texts_path, "texts_original")
            # Rename original data folder
            os.rename(args.texts_path, altered_original_folder)
            # Rename adapted data folder to Pseudo-original data folder
            os.rename(args.adapted_prompts_path, args.texts_path)

        # run t2m model
        try:
        # run t2m model
            if args.model == "mo_mask":
                #eval_file_exec = "eval_t2m_vq.py"
                #exec_file = os.path.join(MOMASK_REPO_DIR, eval_file_exec)
                #dataset_name = "kit" if args.dataset == "kit" else "t2m"
                
                cmd = f'cd external_repos/momask-codes; python eval_t2m_trans_res.py --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --dataset_name t2m --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --gpu_id 0 --cond_scale 4 --time_steps 10 --ext evaluation --batch_size 2 > {logfile.name}'
                # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                os.system(cmd)
                #sys.path.append(MOMASK_REPO_DIR) # add to sys so that modules inside the repo can be imported
                #sys.path.append(os.path.join(MOMASK_REPO_DIR, "utils/"))
                #sys.argv = ['--res_name=tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw_k', f'--dataset_name={dataset_name}', "--name=t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns_k", "--gpu_id=0", "--cond_scale=2", "--time_steps=10", "--ext=evaluation"] 
                
                # exec(open(exec_file).read(), {"__name__": ""})
        except Exception as e: 
            # if there is any error while running the T2M model, we still want to continue this script to rename the folder to their original names 
            logging.error(e)
            print(e)
        finally:
            # rename folders back to their original names if prompt adaptation was performed
            if args.prompt_adaptation_model:
                os.rename(args.texts_path, altered_original_folder)
                # Rename adapted data folder to Pseudo-original data folder
                os.rename(args.adapted_prompts_path, args.texts_path)
        
        logging.info("Finished with the following args:")
        logging.info(args)

    end_logging(logfile)