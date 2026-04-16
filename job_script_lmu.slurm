#!/bin/bash
#SBATCH --job-name=HF4
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

##SBATCH -p inter
##SBATCH --gres=gpu:h100nvl:1

##SBATCH -p inter
##SBATCH --gres=gpu:v100:1

##SBATCH -p cip
##SBATCH --gres=gpu:a40:1

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

##SBATCH --array=0-9

#SBATCH --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

#nvidia-smi

J_VALUES=(0.5)
SEED_VALUES=(1)
#BC_COMBINATIONS=("APC PBC" "APC APC" "PBC APC" "PBC PBC" )
BC_COMBINATIONS=("PBC PBC" )
PHI_VALUES=(0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0) 


NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
NUM_BC=${#BC_COMBINATIONS[@]}
NUM_PHI=${#PHI_VALUES[@]}

TOTAL_JOBS=$(( NUM_J * NUM_SEED * NUM_BC * NUM_PHI ))


BC_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_BC ))
SEED_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_BC) % NUM_SEED ))
J_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_BC * NUM_SEED)) % NUM_J ))
PHI_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_BC * NUM_SEED * NUM_J)) % NUM_PHI ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}
BC_PAIR=(${BC_COMBINATIONS[$BC_INDEX]})
BC_X=${BC_PAIR[0]}
BC_Y=${BC_PAIR[1]}
PHI=${PHI_VALUES[$PHI_INDEX]}



sleep 10
#srun python -m ViT_Heisenberg.run_ViT --L 6 --seed "$SEED" --J2 "$J"
#srun python -m HFDS_Heisenberg.run_spin --L 4 --seed "$SEED" --J2 "$J"
sleep 10

#srun python -m Entanglement.test_var_init
#srun python -m Entanglement.test_var_init_no_bipartite

#srun python -m Observables
#srun python -m Elaborate.Statistics.Degeneracy

srun python -m HFDS_Heisenberg.run_spin_phi --L 4 --seed "$SEED" --J2 "$J" --bc_x "$BC_X" --bc_y "$BC_Y" --phi "$PHI"
