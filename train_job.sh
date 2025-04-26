#!/bin/bash
#SBATCH --account=ACD114028
#SBATCH --partition=normal
#SBATCH --job-name=data_prep
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=2-00:00:00

source /home/twtomtwcc00/VideoBlurRemoval/.venv/bin/activate
/home/twtomtwcc00/.local/bin/uv run /home/twtomtwcc00/VideoBlurRemoval/main2.py