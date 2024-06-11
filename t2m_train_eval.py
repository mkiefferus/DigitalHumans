# Main file to run the evaluation and training of the different T2M models 

import argparse
import subprocess
import logging
import os

from utils import SESSION_ID, MOMASK_REPO_DIR, HUMAN_ML_DIR
from utils.logging import init_logging_old_python_version, end_logging

def parse_args():
    # all args that cannot be matched to the Trainer or DataLoader classes and are
    parser = argparse.ArgumentParser(description='Implementation of ETHZ Digital Humans Project for improving Text to Motion generations by enhancing the text prompt.')
    parser.add_argument('-e', '--experiment_name', type=str, required=False, 
                        help="The experiment name. Default to the enhanced texts name \
                        with an additional value of the current time.")
    parser.add_argument('--texts_folder_name', type=str, required=True,
                        help="The name of the folder containing the texts to be used for training or evaluation. \
                            It should be located in the Momask HumanML3D dataset folder")
    parser.add_argument('-tm', '--train_mask', type=bool, default=False,
                        help="Set to true if you want to train the Masked Transformer end-to-end")
    parser.add_argument('-tr', '--train_res', type=bool, default=False,
                        help="Set to true if you want to train the Residual Transformer end-to-end")
    parser.add_argument('-t', '--eval', type=bool, default=False,
                        help="Set to true if you want to evaluaate the model. If true, specify the model checkpoint.")
    parser.add_argument('--evaluate_single_samples', type=str, required=False,
                        help="Whether to generate single scores for each sample in the dataset.")
    
    parser.add_argument('--model_checkpoint', type=str, required=False, default="original",
                        help="Which model to evaluate. Defaults to the original MoMask model.")
    parser.add_argument('-v', '--verbose', type=bool, required=False, default=False,
                        help="Whether to output information into the console (True) or the logfile (False).")

    known_args, _ = parser.parse_known_args()
    
    # check if experiment name is set, otherwise create automatically
    if known_args.experiment_name is None:
        known_args.experiment_name = f"{known_args.texts_folder_name}_{SESSION_ID}"
    
    # specify the arguments for the experiment, dataloader, t2m-model and prompt adaptation
    
    return known_args
    
if __name__ == '__main__':
    args = parse_args()
    
    # init log file
    logfile, log_file_name = init_logging_old_python_version(args.experiment_name, args.verbose)
    
    # rename the original 'texts' folder to the enhanced texts so that momask trains on the enhanced texts
    args.texts_folder = os.path.join(HUMAN_ML_DIR, args.texts_folder_name)
    altered_original_folder = os.path.join(HUMAN_ML_DIR, "texts_original")
    original_folder = os.path.join(HUMAN_ML_DIR, "texts")
    os.rename(original_folder, altered_original_folder)
    os.rename(args.texts_folder, original_folder)
    
    # Put all in a try except block so that the folders are renamed to their original statusif there is an error
    
    # TODO:
    #     - singular evaluations: change files in momask
    #     - change model names for checkpoints
    #     - train_mask: train masked transformer 
    #     - train_res: train residual transformer
    #     - eval: evaluate model
    try:
        
        print("Here")
        import subprocess
        if args.train_mask:
            # train massked transformer
            # if args.model_checkpoint == "original" and not args.verbose:
            #     cmd = f'cd external_repos/momask-codes;python eval_t2m_trans_res.py --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --dataset_name t2m --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --gpu_id 0 --cond_scale 4 --time_steps 10 --ext evaluation --batch_size 2 > {logfile} 2>&1'
            # # p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            # os.system(cmd)
            command = [
                'python', 'eval_t2m_trans_res.py',
                '--res_name', 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw',
                '--dataset_name', 't2m',
                '--name', 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns',
                '--gpu_id', '0',
                '--cond_scale', '4',
                '--time_steps', '10',
                '--ext', 'evaluation',
                '--batch_size', '2'
            ]
            from pathlib import Path
            # Set the working directory
            working_dir = MOMASK_REPO_DIR

            # Open the log file in write mode
            with open(log_file_name, 'w') as f:
                # Run the command in the specified working directory
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=working_dir)
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
        os.rename(original_folder, args.texts_folder)
        os.rename(altered_original_folder, original_folder)
        
        logging.info("Finished with the following args:")
        logging.info(args)
        end_logging(logfile)