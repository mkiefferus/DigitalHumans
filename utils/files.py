from time import time

import os
import torch

# directories
SESSION_ID = str(time())
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # the project root directory
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXTERNAL_REPOS_DIR = os.path.join(ROOT_DIR, "external_repos")
MOMASK_REPO_DIR = os.path.join(EXTERNAL_REPOS_DIR, "momask-codes")
PROMPT_MODEL_FILES_DIR = os.path.join(DATA_DIR, "prompt_llm_models")
HUMAN_ML_DIR = os.path.join(MOMASK_REPO_DIR, "dataset/HumanML3D")

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if repo is cloned for the first time, create empty folder structure, so no "PATH NOT FOUND" errors occur
for folder in [LOG_DIR, PROMPT_MODEL_FILES_DIR, EXTERNAL_REPOS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder {folder}")