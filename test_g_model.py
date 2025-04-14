# test_rl_policy.py

import os
import torch
import hydra
from omegaconf import DictConfig
from core.runner.runner import set_seed
from core.tasks.rl_task import RLTask
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

@hydra.main(config_path="configs/task", config_name="rl.yaml", version_base='1.2')
def main(cfg: DictConfig):
    # === Set seed ===
    set_seed(cfg.get("seed", 42))

    # === Initialize RL object with Hydra config ===
    #rl = RLTask(cfg)

    base_dir = Path(__file__).resolve().parent
    param_path = base_dir / "param_data/Pendulum-v1/data.pt"
    param_dict = torch.load(param_path)

    #print("param_dict:", param_dict)


    param_data= param_dict['pdata']


    # Compute pairwise L2 distance
    dist_matrix = torch.cdist(param_data, param_data, p=2).cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, cmap="viridis")
    plt.title("Pairwise Distance between Param Data")
    plt.savefig("pairwise_distance.png")


    norms = torch.norm(param_data, dim=1)
    print("Mean Norm:", norms.mean().item())
    print("Std of Norm:", norms.std().item())

if __name__ == "__main__":
    main()