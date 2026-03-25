#!/bin/bash
#SBATCH --job-name=Sign
#SBATCH --output=slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB

##SBATCH --partition=gpupr.4h
##SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1

##SBATCH --gpus=4   
##SBATCH --gres=gpumem:40g


#SBATCH --time=05-00:00:00
##SBATCH --array=0-2

#nvidia-smi

module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load python/3.11.6
source ../gpu_netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu

J_VALUES=(0.5)
SEED_VALUES=(1 2 3)
#BC_COMBINATIONS=("APC PBC" "APC APC" "PBC PBC")
BC_COMBINATIONS=("PBC PBC")


NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
NUM_BC=${#BC_COMBINATIONS[@]}

BC_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_BC ))
J_SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_BC ))
SEED_INDEX=$(( J_SEED_INDEX % NUM_SEED ))
J_INDEX=$(( J_SEED_INDEX / NUM_SEED ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}
BC_PAIR=(${BC_COMBINATIONS[$BC_INDEX]})
BC_X=${BC_PAIR[0]}
BC_Y=${BC_PAIR[1]}


#srun python -m ViT_Heisenberg.run_ViT_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"
#srun python -m HFDS_Heisenberg.run_spin_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"
#sleep 2
#python -m HFDS_Heisenberg.run_spin --L 10 --seed "$SEED" --J2 "$J" --bc_x "$BC_X" --bc_y "$BC_Y"
#python -m ViT_Heisenberg.run_ViT --L 10 --seed "$SEED" --J2 "$J" --bc_x "$BC_X" --bc_y "$BC_Y"
#sleep 2

#srun python -m ED.quspin_MCState
#srun python -m ED.quspin_6x6 --J2 "$J"

srun python -m Observables
#srun python -m Entanglement.test_var_init