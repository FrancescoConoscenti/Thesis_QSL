#!/bin/bash
#SBATCH --job-name=HFDS_ferm
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

##SBATCH --gres=gpu:a40:1
##SBATCH --partition=cip,inter

#SBATCH --mem=16G
#SBATCH --time=12:00:00

sleep 10
export JAX_PLATFORMS=cpu
source /scratch/f/F.Conoscenti/netket_venv1/bin/activate

J_VALUES=(0.0 0.2 0.5 0.7 1.0)
J=${J_VALUES[$SLURM_ARRAY_TASK_ID]}

#srun python HFDS_Heisenberg/run_ferm.py
srun python HFDS_Heisenberg/run_spin.py --J2 $J
#srun python ViT_Heisenberg/run_ViT.py