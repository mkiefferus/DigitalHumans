# script to initialize log files

import logging
import os
import sys
from utils.utils import LOG_DIR, SESSION_ID

def init_logging(experiment_name, run_name):
    log_name = f"{experiment_name}_{run_name}_{SESSION_ID}.log"
    log_file = os.path.join(LOG_DIR, log_name)
    logging.basicConfig(filename=log_file,
                    encoding='utf-8',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
    
    
    file = open(log_file,"w")
    sys.stdout = file
    
    print(f"Experiment: {experiment_name}, Run: {run_name}")
    return file

def end_logging(logfile):
    logfile.close()
    

import numpy as np
import pandas as pd
from time import time
import os
SESSION_ID = str(time())

def track_scores(text, fid, diversity, r_precision, matching_score, multimodality, out_path:str="./"):
    """
    Track the scores of the model
    :param fid: FID score
    :param diversity: Diversity score
    :param r_precision: R-Precision score
    :param matching_score: Matching score
    :param multimodality: Multimodality score
    :param out_path: Path to save the scores
    :return: None
    """
    org_scores_path = "score_log.csv"

    org_scores = pd.read_csv(org_scores_path, delimiter=";")

    scores = pd.DataFrame(
        {   
            "text": [text],
            "fid_pred": [fid],
            "diversity_pred": [diversity],
            "r_precision_pred": [r_precision],
            "matching_score_pred": [matching_score],
            "multimodality_pred": [multimodality],
        }
    )

    # Merge org_scores and new scores
    final_scores = pd.merge(org_scores, scores, on="text", how="outer")

    # Generate output path
    out_path = os.path.join(out_path, f"{SESSION_ID}.csv")

    final_scores.to_csv(out_path, index=False)