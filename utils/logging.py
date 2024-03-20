# script to initialize log files

import logging
import os
from utils import LOG_DIR, SESSION_ID

def init_logging(experiment_name, run_name):
    log_name = f"{experiment_name}_{run_name}_{SESSION_ID}.log"
    logging.basicConfig(filename=os.path.join(LOG_DIR, log_name),
                    encoding='utf-8',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
    logging.log(logging.INFO, f"Experiment: {experiment_name}, Run: {run_name}")
    