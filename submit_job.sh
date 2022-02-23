#!/bin/bash
#SBATCH --job-name=lmpDDQN
#SBATCH --account=AICDI
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate LammpsSB3

rm slurm-*
python -u train.py > MoS2/DQN_log.txt
mv log.lammps MoS2
mv *.png MoS2

