#!/bin/bash
#SBATCH --job-name=HFDS8x8
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

##SBATCH -p inter
##SBATCH --gres=gpu:h100nvl:1

##SBATCH -p inter
##SBATCH --gres=gpu:v100:1

#SBATCH -p cip
#SBATCH --gres=gpu:a40:1

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --mem=32G
#SBATCH --time=02-00:00:00

#SBATCH --array=0-8

#SBATCH --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

nvidia-smi

J_VALUES=(0.5)
SEED_VALUES=(1 2 3)
BC_COMBINATIONS=("APC PBC" "APC APC" "PBC PBC")

NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
NUM_BC=${#BC_COMBINATIONS[@]}

TOTAL_JOBS=$(( NUM_J * NUM_SEED * NUM_BC ))


BC_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_BC ))
J_SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_BC ))
SEED_INDEX=$(( J_SEED_INDEX % NUM_SEED ))
J_INDEX=$(( J_SEED_INDEX / NUM_SEED ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}
BC_PAIR=(${BC_COMBINATIONS[$BC_INDEX]})
BC_X=${BC_PAIR[0]}
BC_Y=${BC_PAIR[1]}



#sleep 10
#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J"
srun python -m HFDS_Heisenberg.run_spin --L 8 --seed "$SEED" --J2 "$J" --bc_x "$BC_X" --bc_y "$BC_Y"
#sleep 10

#srun python -m Entanglement.Entanglement
#srun python -m Entanglement.test_var_init_no_bipartite

#srun python -m Observables