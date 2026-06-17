#!/bin/bash -l
#SBATCH --job-name=qwen7B_ASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct
#SBATCH --time=112:00:00
#SBATCH --chdir=/home/tthebau1/SPEAR/followup7B
#SBATCH --output=/home/tthebau1/SPEAR/followup7B/slurm_output/qwen2.5_vox1-H_%j.log
#SBATCH --error=/home/tthebau1/SPEAR/followup7B/slurm_output/qwen2.5_vox1-H_%j.log

source 
echo "python: $(command -v python3)"
python3 -V

# ----- Paths (edit to your locations) -----
BASE="/home/tthebau1/SPEAR/followup7B"
PIPE="${BASE}/run_ASV_qwen.py"
MODEL="Qwen/Qwen2.5-Omni-7B"
SEED=666
START_OVER=False

# Where to store per-question outputs + kept turnovers
RUN_DIR="${BASE}/runs_asv_evidence"
mkdir -p "${RUN_DIR}"
TRIAL='h'
echo "============================================================"
echo "[INFO] Output dir: ${RUN_DIR}"
echo "[INFO] Trial: ${TRIAL}"
echo "============================================================"


/home/tthebau1/miniconda3/envs/speechllm/bin/python3 -u "${PIPE}" \
--seed "${SEED}" \
--start-over "${START_OVER}" \
--trial "${TRIAL}" \
--qwen-model "${MODEL}" 

