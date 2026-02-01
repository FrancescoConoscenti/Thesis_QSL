#!/bin/bash
#SBATCH --job-name=4x4svd
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1


##SBATCH -p inter
##SBATCH --gres=gpu:v100:1

##SBATCH -p cip
##SBATCH --gres=gpu:a40:1

##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32


#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
##SBATCH --array=0-14

#SBATCH --export=ALL,XLA_PYTHON_CLIENT_PREALLOCATE=false,XLA_PYTHON_CLIENT_MEM_FRACTION=0.85


source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85



J_VALUES=(0.4 0.5 0.55 0.6 0.7)
SEED_VALUES=(1 2 3)

NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}

J_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_J ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_J ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}


sleep 10
#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" --J2 "$J"
#srun python -m HFDS_Heisenberg.run_spin --seed "$SEED" --J2 "$J"
sleep 10
#srun python -m Entanglement.Entanglement
#srun python -m Entanglement.test_var_init


srun python -m Observables