#!/bin/bash

# Check if current directory is DigitalHumans
if [[ $(basename "$(pwd)") != "DigitalHumans" ]]; then
    echo "Please navigate to the DigitalHumans directory first."
    exit 1
fi

# Directory where repositories will be cloned
repos_dir="$(pwd)/external_repos"

# Check if external_repos directory exists
if [ ! -d "$repos_dir" ]; then
    echo "Error: The 'external_repos' directory does not exist."
    exit 1
fi

echo "Cloning necessary repositories ..."

# Clone momask-codes repository
repo_name=$(basename https://github.com/EricGuo5513/momask-codes .git)

if [ ! -d "$repos_dir/$repo_name" ]; then
    git clone https://github.com/EricGuo5513/momask-codes "$repos_dir/$repo_name"
    rm -rf "$repos_dir/$repo_name/.git"
else
    echo "Repository '$repo_name' already exists"
fi

echo "Setup complete"

