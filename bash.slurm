#!/bin/sh

#SBATCH --job-name=="superconducting_test"
#SBATCH --account=education-eemcs-msc-am
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module load cuda/12.21
module load python
module load miniconda3
module load 2023r1
module load openmpi
module load py-torch

conda activate /home/aheuvel/aheuvelvenv

srun python get_energies_numpy.py