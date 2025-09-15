#!/bin/bash
#SBATCH --job-name=ground_state
#SBATCH --output=ViT.out
#SBATCH --error=ViT.txt
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load python/3.11.6
source ~/netket_venv/bin/activate

# Force JAX to CPU
export JAX_PLATFORMS=cpu

srun python HFDS_Heisenberg/run.py
#srun python ViT_Heisenberg/ground_state.py



