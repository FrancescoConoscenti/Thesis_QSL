#!/bin/bash
#SBATCH --job-name=Vit6x6
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:1

#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-5

source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

export JAX_PLATFORMS=cpu


J_VALUES=( 0.55 0.6)
SEED_VALUES=(1 2 3)

NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}

J_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_J ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_J ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}


srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J"
#srun python -m HFDS_Heisenberg.run_spin --seed "$SEED" --J2 "$J"
