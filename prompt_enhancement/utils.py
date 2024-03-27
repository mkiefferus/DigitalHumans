from time import time

import os
import torch

# directories
SESSION_ID = str(time())
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # the project root directory
LOG_DIR = os.path.join(CURRENT_DIR, "out", "logs")
DATA_DIR = os.path.join(CURRENT_DIR, "data", "datasets")
T2M_MODEL_FILES_DIR = os.path.join(CURRENT_DIR, "data", "t2m_model_weights")
PROMPT_MODEL_FILES_DIR = os.path.join(CURRENT_DIR, "data", "prompt_llm_models")
OUT_DIR = os.path.join(CURRENT_DIR, "out")

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"