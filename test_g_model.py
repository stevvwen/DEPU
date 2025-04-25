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
import torch.nn.functional as F

@hydra.main(config_path="configs/task", config_name="rl.yaml", version_base='1.2')
def main(cfg: DictConfig):
    # === Set seed ===
    set_seed(cfg.get("seed", 42))

    # === Load saved parameter data ===
    base_dir = Path(__file__).resolve().parent
    param_path = base_dir / "param_data/Pendulum-v1/data.pt"
    param_dict = torch.load(param_path)
    param_data = param_dict['pdata']  # Shape: [N, D]

    print("Parameter data shape:", param_data.shape)

    # === Optionally sort each param vector (optional step, can be removed if unnecessary) ===
    sorted_param_data = []
    for data in param_data:
        data, _ = torch.sort(data)
        sorted_param_data.append(data)
    sorted_param_data = torch.stack(sorted_param_data)

    # === Compute cosine similarity matrix ===
    normed = F.normalize(param_data[:200], p=2, dim=1)  # Normalize rows
    dist_matrix = torch.mm(normed, normed.T).cpu().numpy()  # [200, 200] cosine similarity

    # === Plot heatmap ===
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, cmap="viridis")
    plt.title("Pairwise Cosine Similarity between Param Data")
    plt.savefig("pairwise_cosine_similarity.png")
    plt.close()

    # === Norm statistics ===
    norms = torch.norm(param_data, dim=1)
    print("Mean Norm:", norms.mean().item())
    print("Std of Norm:", norms.std().item())

if __name__ == "__main__":
    main()
