#!/bin/bash
#SBATCH --job-name=HFDS_spin
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32
##SBATCH --gres=gpu:1

#SBATCH --mem=64G
#SBATCH --time=00-24:00:00
##SBATCH --array=0-15

source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu

J_VALUES=(0.2 0.5 0.7 1)
SEED_VALUES=(1 2 3 4)


NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}

J_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_J ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_J ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}


#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J"
#srun python -m HFDS_Heisenberg.run_ferm --J2 $J
#srun python -m HFDS_Heisenberg.run_spin --seed "$SEED" --J2 "$J"
srun python -m Elaborate.QSL_DMRG