#!/bin/bash
#SBATCH --job-name=HFDS_spin_true
#SBATCH --output=slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8
##SBATCH --gres=gpu:1

#SBATCH --time=00-1:00:00
##SBATCH --array=0-14

source ../netket_venv/bin/activate

export JAX_PLATFORMS=cpu

J_VALUES=(0.0 0.2 0.5 0.7 1.0)
SEED_VALUES=(1 2 3)

NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}

J_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_J ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_J ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}


#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J"
#srun python -m HFDS_Heisenberg.run_spin --seed "$SEED" --J2 "$J"

#srun python -m Elaborate.Plotting.plot_avarage_models
srun python -m Elaborate.Plotting.Fermi_Init_fidelity_vs_Js

