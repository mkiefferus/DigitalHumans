from time import time

import os
import torch

# directories
SESSION_ID = str(time())
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
LOG_DIR = os.path.join(ROOT_DIR, "out", "logs")
OUT_DIR = os.path.join(ROOT_DIR, "out")
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXTERNAL_REPOS_DIR = os.path.join(ROOT_DIR, "external_repos")
MOMASK_REPO_DIR = os.path.join(EXTERNAL_REPOS_DIR, "momask-codes")
PROMPT_MODEL_FILES_DIR = os.path.join(DATA_DIR, "prompt_llm_models")


# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")