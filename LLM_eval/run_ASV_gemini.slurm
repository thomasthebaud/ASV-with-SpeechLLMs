#!/bin/bash -l
#SBATCH --job-name=gemini_ASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --time=112:00:00
#SBATCH --chdir=/home/tthebau1/SPEAR/followup7B
#SBATCH --output=/home/tthebau1/SPEAR/followup7B/slurm_output/gemini2.5-vox1-h.log
#SBATCH --error=/home/tthebau1/SPEAR/followup7B/slurm_output/gemini2.5-vox1-h.log

/home/tthebau1/miniconda3/envs/speechllm/bin/python3 -u run_ASV_gemini.py --trial 'h'

