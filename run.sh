#!/bin/sh

#SBATCH --job-name="superconducting_test"
#SBATCH -e stderrs/%j-errors-training.err
#SBATCH -o stdouts/%j-outputs-training.out
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=education-eemcs-bsc-tw

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate gpu_env

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

srun python get_energies_numpy.py

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate