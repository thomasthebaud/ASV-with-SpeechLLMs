#!/bin/bash -l
#SBATCH --job-name=Kimi_ASV
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --chdir=/home/tthebau1/SPEAR/followup7B
#SBATCH --output=/home/tthebau1/SPEAR/followup7B/slurm_output/ASV_all_%j.log
#SBATCH --error=/home/tthebau1/SPEAR/followup7B/slurm_output/ASV_all_%j.log

# /home/tthebau1/miniconda3/envs/speechllm/bin/python3 -u run_ASV_all.py

export LD_LIBRARY_PATH=$HOME/miniconda3/envs/SPEAR2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/slurm_files/software/cuda/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
conda activate SPEAR2

python -c "import ctypes; ctypes.CDLL('libcudart.so.12'); print('cudart ok')"
python -c "import flash_attn; print('flash ok')"
which python
python -c "import torch; import flash_attn; print(torch.__version__, flash_attn.__version__)"

echo "starting the job"
/home/tthebau1/miniconda3/envs/SPEAR2/bin/python3 -u run_ASV_all.py