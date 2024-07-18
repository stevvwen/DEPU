#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --output=scripts/out2.txt
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64000M

module load python/3.10
module load cuda

source ../depu/bin/activate

python test_td3_training.py