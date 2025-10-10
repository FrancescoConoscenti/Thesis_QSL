#!/bin/bash
#SBATCH --job-name=HFDS_ferm
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

##SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1
##SBATCH --partition=cip

#SBATCH --mem=16G
#SBATCH --time=00-12:00:00

source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

export JAX_PLATFORMS=cpu

J_VALUES=(0.0 0.2 0.5 0.7 1.0)
J=${J_VALUES[$SLURM_ARRAY_TASK_ID]}

#srun python -m HFDS_Heisenberg.run_ferm --J2 $J
srun python -m HFDS_Heisenberg.run_spin --J2 $J
#srun python -m ViT_Heisenberg.run_ViT --J2 $J