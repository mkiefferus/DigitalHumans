from .utils import *
from .logging import *
from .download import *

# if repo is cloned for the first time, create empty folder structure, so no "PATH NOT FOUND" errors occur
import os
for folder in [LOG_DIR, PROMPT_MODEL_FILES_DIR, EXTERNAL_REPOS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder {folder}")