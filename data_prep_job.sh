#!/bin/bash
#SBATCH --account=ACD114028
#SBATCH --partition=normal
#SBATCH --job-name=data_prep
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

source /home/twtomtwcc00/VideoBlurRemoval/.venv/bin/activate
/home/twtomtwcc00/.local/bin/uv run /home/twtomtwcc00/VideoBlurRemoval/data_prep.py
/home/twtomtwcc00/.local/bin/uv run /home/twtomtwcc00/VideoBlurRemoval/data_prep_blurred.py