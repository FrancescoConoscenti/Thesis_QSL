#!/bin/bash
#SBATCH --job-name=10x104ksamples
#SBATCH --output=slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB

##SBATCH --partition=gpupr.4h
##SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

#SBATCH --gpus=1
#SBATCH --gres=gpumem:60g

#SBATCH --time=4-00:00:00
#SBATCH --array=0-5

nvidia-smi

module load stack/2024-06
module load gcc/12.2.0
module load openmpi/4.1.6
module load python/3.11.6
source ../gpu_netket_venv1/bin/activate

#export JAX_PLATFORMS=cpu

J_VALUES=(0.5 0.55 0.6)
SEED_VALUES=(1 2)
OUTPUT_VALUES=(psi exact_sign)


NUM_J=${#J_VALUES[@]}
NUM_SEED=${#SEED_VALUES[@]}
NUM_OUTPUT=${#OUTPUT_VALUES[@]}

SEED_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_SEED ))
J_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_SEED) % NUM_J ))
OUTPUT_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_SEED * NUM_J)) % NUM_OUTPUT ))

J=${J_VALUES[$J_INDEX]}
SEED=${SEED_VALUES[$SEED_INDEX]}
OUTPUT=${OUTPUT_VALUES[$OUTPUT_INDEX]}


#srun python -m ViT_Heisenberg.run_ViT_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"
#srun python -m HFDS_Heisenberg.run_spin_amp --seed "$SEED" --J2 "$J" --output "$OUTPUT"
sleep 10
python -m HFDS_Heisenberg.run_spin --seed "$SEED" --J2 "$J"
sleep 10
#srun python -m try_gpus
#srun python -m ViT_Heisenberg.run_ViT --seed "$SEED" #--J2 "$J" 

#srun python -m ED.quspin_MCState
#srun python -m ED.quspin_6x6 --J2 "$J"

#srun python -m Observables

#srun python -m DMRG.DMRG_test_Imp_sampl --J2 "$J"

#srun python -m HFDS_Heisenberg.test_Init_grid_search
#srun python -m ED.quspin_6x6

#srun python -m DMRG.Observables_Fidelity --J2 "$J"