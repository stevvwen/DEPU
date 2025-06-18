import os
import glob
import shutil
import json
import datetime
import torch
from joblib import Parallel, delayed

from data.agent_trainer import AgentTrainer
from utils import (
    get_storage_usage,
    extract_agent_params,
    config_to_dict,
)

class RLTask:
    def __init__(self, cfg):
        self.cfg            = cfg
        self.trainer        = AgentTrainer(cfg)
        self.tmp_time       = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        # for naming saved data
        self.cfg.data.dataset = cfg.rl_env.env_name
        self.num_agents     = cfg.num_agents
        self.visualization  = not cfg.debug_mode and cfg.eval_mode

        # decide which layers to train
        if cfg.train_layer == "all":
            self.actor_layers  = [
                name for name, _ in self.trainer.agent.actor.named_parameters()
            ]
            self.critic_layers = [
                name for name, _ in self.trainer.agent.critic.named_parameters()
            ]
        else:
            self.actor_layers  = cfg.train_layer.actor
            self.critic_layers = cfg.train_layer.critic


    def train_for_data(self):
        """
        Train multiple agents in parallel, save their params, and aggregate results.
        """
        save_root = getattr(self.cfg, "save_root", "param_data")
        tmp_dir   = os.path.join(save_root, f"tmp_{self.tmp_time}")
        final_dir = os.path.join(save_root, self.cfg.data.dataset)
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        # train agents concurrently
        scores = Parallel(n_jobs=16)(
            delayed(self._train_one)(i, tmp_dir)
            for i in range(self.num_agents)
        )

        # aggregate parameters
        batch, mean, std = self._aggregate(tmp_dir)
        usage = get_storage_usage(tmp_dir)
        print(f"Storage for {tmp_dir}: {usage:.2f} GB")

        save_state = {
            "pdata":       batch.cpu(),
            "mean":        mean.cpu(),
            "std":         std.cpu(),
            "train_layer": self.actor_layers + self.critic_layers,
            "performance": scores,
            "cfg":         config_to_dict(self.cfg),
        }
        torch.save(save_state, os.path.join(final_dir, "data.pt"))
        json.dump(
            {"cfg": config_to_dict(self.cfg), "performance": scores},
            open(os.path.join(final_dir, "config.json"), "w")
        )

        shutil.rmtree(tmp_dir)
        print("Data processing completed.")
        return {"save_path": final_dir}

    def _train_one(self, agent_id, tmp_dir):
        """Train one agent and save its parameters."""
        trainer = AgentTrainer(self.cfg)
        trainer.rollout()
        avg_score, _ = trainer.evaluate()
        params = extract_agent_params(
            self.actor_layers, self.critic_layers, trainer.agent
        )
        save_file = os.path.join(tmp_dir, f"p_data_{agent_id}.pt")
        torch.save(params, save_file)
        print(f"Agent {agent_id} done: {avg_score}")
        return avg_score

    def _aggregate(self, tmp_dir):
        """Load saved params and compute batch, mean, std."""
        pdata = []
        for fn in glob.glob(os.path.join(tmp_dir, "p_data_*.pt")):
            buf = torch.load(fn)
            chunks = [buf[k].reshape(-1) for k in buf if k in self.actor_layers + self.critic_layers]
            pdata.append(torch.cat(chunks))
        batch = torch.stack(pdata)
        return batch, batch.mean(0), batch.std(0)

