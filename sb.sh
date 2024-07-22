#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --output=scripts/out3.txt
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M

module load python/3.11

source ../depu/bin/activate

python -m rl_zoo3.train --algo td3 --env Pendulum-v1 --n-timesteps 1000000 --log-interval 10 --eval-freq 1000
python -m rl.zoo3.scripts/plot_train.py -a td3 -e Pendulum-v1 -y success -f rl-trained-agents/ -w 500 -x steps