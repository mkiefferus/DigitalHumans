import argparse
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # the project root directory
HUMAN_ML_DIR = os.path.join(ROOT_DIR, "external_repos/momask-codes/dataset/HumanML3D")

def partition_file(input_file, num_partitions):
    # Read the input file
    with open(os.path.join(HUMAN_ML_DIR, input_file), 'r') as file:
        lines = file.readlines()
    
    # Calculate the size of each partition
    total_lines = len(lines)
    partition_size = total_lines // num_partitions
    remainder = total_lines % num_partitions

    # Create partitions
    start = 0
    for i in range(num_partitions):
        end = start + partition_size + (1 if i < remainder else 0)
        partition_lines = lines[start:end]
        
        # Write the partition to a new file
        output_file = f'partition_{i:03}.txt'
        with open(os.path.join(HUMAN_ML_DIR, output_file), 'w') as file:
            file.writelines(partition_lines)
        
        start = end
        print(f'Created {output_file} with {len(partition_lines)} lines.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Partition a text file into specified number of new files.')
    parser.add_argument('--input_file', type=str, help='The input text file to partition.', required=True)
    parser.add_argument('--num_partitions', type=int, help='The number of partitions to create.', required=True)

    args = parser.parse_args()
    input_file = args.input_file
    num_partitions = args.num_partitions

    if not os.path.isfile(os.path.join(HUMAN_ML_DIR, input_file)):
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)
    
    if num_partitions <= 0:
        print("Error: Number of partitions must be a positive integer.")
        sys.exit(1)
    
    partition_file(input_file, num_partitions)
