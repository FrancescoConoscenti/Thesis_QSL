#!/bin/bash
#SBATCH --job-name=ground_state  # Job name
#SBATCH --output=ViT.out
#SBATCH --error=ViT.txt
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpumem:8g
#SBATCH --partition=gpu

c 'import jax; print(jax.devices())'

module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load python/3.11.6

source ~/gpu_netket_venv/bin/activate
srun python ViT_Heisenberg/ground_state.py


