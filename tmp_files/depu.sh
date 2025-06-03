#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --output=scripts/out4.txt
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=54000M

module load python/3.11

source ../depu/bin/activate

python task_training.py