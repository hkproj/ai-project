#!/bin/bash

# This script will export the requirements.txt file from the virtual environment
# to the project root directory.

# Export conda's environment into conda-spec-file.txt
conda list --explicit > conda-spec-file.txt

# Export pip requirements into requirements.txt
pip list --format=freeze > pip-requirements.txt