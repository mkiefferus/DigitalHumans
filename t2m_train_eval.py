# Main file to run the evaluation and training of the different T2M models 

import argparse
import subprocess
import logging
import os
import sys

from utils import SESSION_ID, MOMASK_REPO_DIR, HUMAN_ML_DIR
from utils.logging import init_logging_old_python_version, end_logging

def parse_args():
    # all args that cannot be matched to the Trainer or DataLoader classes and are
    parser = argparse.ArgumentParser(description='Implementation of ETHZ Digital Humans Project for improving Text to Motion generations by enhancing the text prompt.')
    parser.add_argument('-e', '--experiment_name', type=str, required=False, 
                        help="The experiment name. Default to the enhanced texts name \
                        with an additional value of the current time.")
    parser.add_argument('-tm', '--train_mask', action="store_true", required=False,
                        help="Set to true if you want to train the Masked Transformer end-to-end")
    parser.add_argument('-tr', '--train_res', action="store_true",
                        help="Set to true if you want to train the Residual Transformer end-to-end")
    parser.add_argument('-t', '--eval_all_metrics', action="store_true", required=False,
                        help="Whether to evaluate the model on all samples in total to get all metrics.")
    parser.add_argument('--eval_single_samples', action="store_true", required=False,
                        help="Whether to generate a multimodal distance scores for each sample in the dataset.")
    parser.add_argument('-v', '--verbose', action="store_true", required=False,
                        help="Whether to output information into the console (True) or the logfile (False).")
    parser.add_argument('-r', '--resume_training',  action="store_true", required=False,
                        help="Whether to resume a training that stopped before.")
    
    # which models and texts to use
    parser.add_argument('--texts_folder_name', type=str, required=True,
                        help="The name of the folder containing the texts to be used for training or evaluation. \
                            It should be located in the Momask HumanML3D dataset folder")
    parser.add_argument('--res_name', type=str, required=False, default="original",
                        help="Which Residual Transformer model to evaluate. Defaults to the original MoMask model.")
    parser.add_argument('--mask_name', type=str, required=False, default="original",
                        help="Which Masked Transformer model to evaluate. Defaults to the original MoMask model.")
    
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
    try:
        res_name = 'tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw' if args.res_name == "original" else args.res_name
        mask_name = 't2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns' if args.mask_name == "original" else args.mask_name
        rvq_name = 'rvq_nq6_dc512_nc512_noshare_qdp0.2'
        # Set the working directory for executing the momask-scripts from
        working_dir = MOMASK_REPO_DIR
        
        # create a Streamdirector to capture all print Statements to the log file if verbose=False:
        class StreamRedirector:
            def __init__(self, log_file):
                self.log_file = log_file
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr

            def __enter__(self):
                self.log_file_handle = open(self.log_file, 'a')
                sys.stdout = self.log_file_handle
                sys.stderr = self.log_file_handle

            def __exit__(self, exc_type, exc_value, traceback):
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                self.log_file_handle.close()
        
        # Train the Masked Transformer
        if args.train_mask:
            command = command = [
                'python', 'train_t2m_transformer.py',
                '--name', mask_name,
                '--gpu_id', '0',
                '--dataset_name', 't2m',
                '--batch_size', '64',
                '--vq_name', rvq_name
            ]
            if args.resume_training:
                command.append('--is_continue')
            if not args.verbose:
                with StreamRedirector(log_file_name):
                    # Run the command in the specified working directory
                    subprocess.run(command, cwd=working_dir)
            else:
                subprocess.run(command, cwd=working_dir)
            
        # Train the Residual Transformer
        if args.train_res:
            command = [
                'python', 'eval_t2m_trans_res.py',
                '--res_name', res_name,
                '--dataset_name', 't2m',
                '--name', mask_name,
                '--gpu_id', '0',
                '--cond_scale', '4',
                '--time_steps', '10',
                '--ext', 'evaluation',
                '--batch_size', '2'
            ]
            if args.resume_training:
                command.append('--is_continue')
            if not args.verbose:
                with StreamRedirector(log_file_name):
                    # Run the command in the specified working directory
                    subprocess.run(command, cwd=working_dir)
            else:
                subprocess.run(command, cwd=working_dir)
        
        # Evaluate the models on all samples to get all metrics
        if args.eval_all_metrics:
            command = [
                'python', 'eval_t2m_trans_res.py', 
                '--res_name', res_name, 
                '--dataset_name', 't2m', 
                '--name', mask_name, 
                '--gpu_id', '0', 
                '--cond_scale', '4', 
                '--time_steps', '10', 
                '--ext', 'evaluation'
            ]
            if not args.verbose:
                with StreamRedirector(log_file_name):
                    # Run the command in the specified working directory
                    subprocess.run(command, cwd=working_dir)
            else:
                subprocess.run(command, cwd=working_dir)
        
        # Evaluate the models to get sample-wise multimodal distances
        if args.eval_single_samples:
            
            # exchange the original momask files with the ones in "adapted codes" for single sample evaluation
            adapted_codes_folder = os.path.join(MOMASK_REPO_DIR, "adapted_codes")
            adapted_dataset_motion_loader = os.path.join(adapted_codes_folder, "dataset_motion_loader.py")
            adapted_eval_t2m = os.path.join(adapted_codes_folder, "eval_t2m.py")
            adapted_get_opt = os.path.join(adapted_codes_folder, "get_opt.py")
            adapted_t2m_dataset = os.path.join(adapted_codes_folder, "t2m_dataset.py")
            adapted_word_vectorizer = os.path.join(adapted_codes_folder, "word_vectorizer.py")
            
            original_dataset_motion_loader = os.path.join(MOMASK_REPO_DIR, "motion_loaders", "dataset_motion_loader.py")
            original_eval_t2m  = os.path.join(MOMASK_REPO_DIR, "utils", "eval_t2m.py")
            original_get_opt  = os.path.join(MOMASK_REPO_DIR, "utils", "get_opt.py")
            original_t2m_dataset  = os.path.join(MOMASK_REPO_DIR, "data", "t2m_dataset.py")
            original_word_vectorizer  = os.path.join(MOMASK_REPO_DIR, "utils", "word_vectorizer.py")
            
            altered_original_dataset_motion_loader = os.path.join(MOMASK_REPO_DIR, "motion_loaders", "dataset_motion_loader_original.py")
            altered_original_eval_t2m  = os.path.join(MOMASK_REPO_DIR, "utils", "eval_t2m_original.py")
            altered_original_get_opt  = os.path.join(MOMASK_REPO_DIR, "utils", "get_opt_original.py")
            altered_original_t2m_dataset  = os.path.join(MOMASK_REPO_DIR, "data", "t2m_dataset_original.py")
            altered_original_word_vectorizer  = os.path.join(MOMASK_REPO_DIR, "utils", "word_vectorizer_original.py")
            
            # rename original files
            os.rename(original_dataset_motion_loader, altered_original_dataset_motion_loader)
            os.rename(original_eval_t2m, altered_original_eval_t2m)
            os.rename(original_get_opt, altered_original_get_opt)
            os.rename(original_t2m_dataset, altered_original_t2m_dataset)
            os.rename(original_word_vectorizer, altered_original_word_vectorizer)
            
            # move adapted files to original file locations
            os.rename(adapted_dataset_motion_loader, original_dataset_motion_loader)
            os.rename(adapted_eval_t2m, original_eval_t2m)
            os.rename(adapted_get_opt, original_get_opt)
            os.rename(adapted_t2m_dataset, original_t2m_dataset)
            os.rename(adapted_word_vectorizer, original_word_vectorizer)
            
            command = [
                'python', 'eval_t2m_trans_res.py', 
                '--res_name', res_name, 
                '--dataset_name', 't2m', 
                '--name', mask_name, 
                '--gpu_id', '0', 
                '--cond_scale', '4', 
                '--time_steps', '10', 
                '--ext', 'evaluation'
            ]
            try:
                if not args.verbose:
                    with StreamRedirector(log_file_name):
                        # Run the command in the specified working directory
                        subprocess.run(command, cwd=working_dir)
                else:
                    subprocess.run(command, cwd=working_dir)
            except Exception as e:
                print(e)
            finally:
                # rename the altered files back to their original names
                os.rename(original_dataset_motion_loader, adapted_dataset_motion_loader)
                os.rename(original_eval_t2m, adapted_eval_t2m)
                os.rename(original_get_opt, adapted_get_opt)
                os.rename(original_t2m_dataset, adapted_t2m_dataset)
                os.rename(original_word_vectorizer, adapted_word_vectorizer)
                
                # rename the original files back to their original names
                os.rename(altered_original_dataset_motion_loader, original_dataset_motion_loader)
                os.rename(altered_original_eval_t2m, original_eval_t2m)
                os.rename(altered_original_get_opt, original_get_opt)
                os.rename(altered_original_t2m_dataset, original_t2m_dataset)
                os.rename(altered_original_word_vectorizer, original_word_vectorizer) 
                               
                
    except Exception as e: 
        # if there is any error while running the T2M model, we still want to continue this script to rename the texts folders to their original names 
        print(e)
    
    finally:
        # rename text folders back to their original names if prompt adaptation was performed
        os.rename(original_folder, args.texts_folder)
        os.rename(altered_original_folder, original_folder)
        
        logging.info("Finished with the following args:")
        logging.info(args)
        end_logging(logfile)