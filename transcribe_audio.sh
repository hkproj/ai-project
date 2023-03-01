#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ai-project
python -c "import os; import numpy; os.system('which whisperx')"
whisperx $1 --model medium --language it --align_model VOXPOPULI_ASR_BASE_10K_IT --output_dir $2