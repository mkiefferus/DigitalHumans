import os
from tqdm import tqdm
import argparse
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
EXTERNAL_REPOS_DIR = os.path.join(ROOT_DIR, "external_repos")
MOMASK_REPO_DIR = os.path.join(EXTERNAL_REPOS_DIR, "momask-codes")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HUMAN_ML_DIR = os.path.join(MOMASK_REPO_DIR, "dataset", "HumanML3D")




def main():
    parser = argparse.ArgumentParser(description="Check dataset quality and handle faulty files.")
    parser.add_argument("--data", type=str, help="Path to the dataset to be checked.")
    parser.add_argument("-f", "--analyse_full_dataset", action="store_false", 
                        help="Analyse the full dataset")
    args = parser.parse_args()

    texts_path = os.path.join(args.data, "texts")

    
    if not args.data:
        args.data = HUMAN_ML_DIR

    if not args.analyse_full_dataset:
        test_txt_path = os.path.join(args.data, "test.txt")
        
        # Load data
        with open(test_txt_path, "r") as f:
            test_files = f.readlines()
            test_files = [file.strip() for file in test_files]

        print(test_files[:10])
    else:
        pass
        


if __name__ == "__main__":
    main()