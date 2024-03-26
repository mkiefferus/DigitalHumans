from time import time
import os

# directories
SESSION_ID = str(time())

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # the project root directory
# CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
print(SESSION_ID)
print(CURRENT_DIR)