#!/bin/bash -l
#SBATCH --job-name=Kimi_ASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --chdir=/home/tthebau1/SPEAR/followup7B
#SBATCH --output=/home/tthebau1/SPEAR/followup7B/slurm_output/ASV_all_%j.log
#SBATCH --error=/home/tthebau1/SPEAR/followup7B/slurm_output/ASV_all_%j.log

which python
/home/ywang792/micromamba/envs/SPEAR/bin/python3 -u run_ASV_all.py