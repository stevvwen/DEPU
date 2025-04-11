# test_rl_policy.py

import os
import torch
import hydra
from omegaconf import DictConfig
from core.runner.runner import set_seed
from core.tasks.rl_task import RLTask
from pathlib import Path

@hydra.main(config_path="configs/task", config_name="rl.yaml", version_base='1.2')
def main(cfg: DictConfig):
    # === Set seed ===
    set_seed(cfg.get("seed", 42))

    # === Initialize RL object with Hydra config ===
    rl = RLTask(cfg)

    base_dir = Path(__file__).resolve().parent
    param_path = base_dir / "param_data/Pendulum-v1/data.pt"
    param_dict = torch.load(param_path)

    print("param_dict:", param_dict)

    print(param_dict['train_layer'])

    param_data= param_dict['pdata']
    
    #for param in param_data:
    #    print(rl.test_g_model(param))

if __name__ == "__main__":
    main()