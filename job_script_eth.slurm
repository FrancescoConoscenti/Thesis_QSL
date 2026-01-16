#!/bin/bash
#SBATCH --job-name=ED
#SBATCH --output=slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=256GB
##SBATCH --gres=gpu:1

#SBATCH --time=00-24:00:00
#SBATCH --array=0-18

source ../netket_venv/bin/activate

export JAX_PLATFORMS=cpu

J_VALUES=(0.0 0.4 0.47 0.50 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.7 1.0)
SEED_VALUES=(1)
OUTPUT_VALUES=(psi exact_sign)


NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
NUM_OUTPUT=${#OUTPUT_VALUES[@]}

J_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_J ))
OUTPUT_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_J) % NUM_OUTPUT ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_J * NUM_OUTPUT)) % NUM_SEED ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}
OUTPUT=${OUTPUT_VALUES[$OUTPUT_INDEX]}


#srun python -m ViT_Heisenberg.run_ViT_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"
#srun python -m HFDS_Heisenberg.run_spin_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"

#srun python -m HFDS_Heisenberg.run_spin #--seed "$SEED" --J2 "$J"
#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J" 

#srun python -m ED.quspin_MCState
srun python -m ED.quspin_6x6 --J2 "$J"

#srun python -m Observables

#srun python -m DMRG.DMRG_test_Imp_sampl --J2 "$J"

#srun python -m HFDS_Heisenberg.test_Init_grid_search
#srun python -m ED.quspin_6x6

#srun python -m DMRG.Observables_Fidelity --J2 "$J"