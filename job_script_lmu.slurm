#!/bin/bash
#SBATCH --job-name=HFDS_ferm
#SBATCH --output=/scratch/f/F.Conoscenti/Thesis_QSL/slurm_out/slurm.%j.%N.out
#SBATCH --ntasks=1

##SBATCH --gres=gpu:a40:1
##SBATCH --partition=cip,inter

#SBATCH --mem=16G
#SBATCH --time=1:00:00


source /scratch/f/F.Conoscenti/netket_venv/bin/activate


#srun python HFDS_Heisenberg/run_ferm.py
#srun python HFDS_Heisenberg/run_spin.py
srun python ViT_Heisenberg/ground_state.py