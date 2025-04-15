import shutil
from .base_task import BaseTask
import glob
import json
import datetime
import wandb
from core.data.agent_trainer import AgentTrainer
from core.data.rl_data import RLData
from joblib import Parallel, delayed
import os
import torch
from core.utils.utils import get_storage_usage, extract_agent_params, replace_agent
from core.utils.format import config_to_dict


def train_one_agent(agent_id, cfg, actor_layers, critic_layers, tmp_path):
    print(f"Training agent {agent_id}...")
    trainer = AgentTrainer(cfg)
    trainer.reset_trainer()
    trainer.rollout()
    avg_score, _ = trainer.evaluate()

    save_path = os.path.join(tmp_path, f"p_data_{agent_id}.pt")
    torch.save(
        extract_agent_params(actor_layers, critic_layers, trainer.agent),
        save_path
    )
    print(f"Agent {agent_id} training finished, average score: {avg_score}")
    return avg_score


class RLTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(RLTask, self).__init__(config, **kwargs)

        self.tmp_time= datetime.datetime.now().strftime('%y%m%d_%H%M%S')

        self.trainer= AgentTrainer(config)

        self.agent_config= config.agent.agent

        self.cfg= config

        # Specify the rl env to be used
        rl_env = str(config.rl_env.env_name)
        self.cfg.data.dataset= rl_env

        self.num_agents= config.num_agents

        self.visualization = not config.debug_mode and config.eval_mode
        

        if self.cfg.train_layer== 'all':
            self.actor_training_layers= [name for name, module in self.trainer.agent.actor.named_parameters()]
            self.critic_training_layers= [name for name, module in self.trainer.agent.critic.named_parameters()]
        else:
            self.actor_training_layers= self.cfg.train_layer.actor
            self.critic_training_layers= self.cfg.train_layer.critic
    

    def set_param_data(self):
        param_data= RLData(self.cfg.param)
        return param_data

    # override the abstract method in base_task.py
    def plot(self, agent_cfg):
        if self.visualization:
            wandb.login()

            run = wandb.init(
                # Set the project where this run will be logged
                project="DEPU",
                name= self.tmp_time,
                config={"learning rate": agent_cfg.lr,
                        "observation size": agent_cfg.obs_shape,
                        "action size": agent_cfg.act_shape,
                        "hidden size": agent_cfg.hidden_dim,
                        },
                allow_val_change=True
            )

            # define our custom x axis metric
            wandb.define_metric("custom_step", hidden=True)
            wandb.define_metric("epi_count", hidden=True)
            wandb.define_metric("eval_epi_count", hidden=True)

            # define which metrics will be plotted against it
            wandb.define_metric("Avg Reward", step_metric="custom_step")
            wandb.define_metric("Epi Reward", step_metric="epi_count")
            wandb.define_metric("Eval Epi Reward", step_metric="eval_epi_count")
    
    

    def test_g_model(self, input):
        param= input
        turns= 20

        test_agent, test_env = self.trainer.set_up_test(config= self.cfg)

        # check the number of parameters
        actor_num= 0
        for name, module in test_agent.actor.named_parameters():
            if name in self.actor_training_layers:
                actor_num += torch.numel(module)

        critic_num= 0
        for name, module in test_agent.critic.named_parameters():
            if name in self.critic_training_layers:
                critic_num += torch.numel(module)

        params_num = torch.squeeze(param).shape[0]
        assert (actor_num+ critic_num == params_num)

        param= torch.squeeze(param).to(test_agent.device)
        
        test_agent= replace_agent(param, test_agent, self.actor_training_layers, self.critic_training_layers, actor_num, critic_num)

        total_score = 0
        test_epi_steps = 0
        for _ in range(turns):
            state, _ = test_env.reset()

            done = False

            while not done:
                action = test_agent.act(state, test_agent.num_expl_steps, True)  # Select action

                next_state, reward, terminated, truncated, info = test_env.step(action)  # Step
                done = terminated or truncated
                total_score += reward

                test_epi_steps += 1
                state = next_state

        avg_score = total_score / turns
        avg_step = test_epi_steps / turns

        return avg_score, avg_step, _

    def val_g_model(self, input):
        #TODO:
        pass

    def train_for_data(self):
        data_path = getattr(self.cfg, 'save_root', 'param_data')
        tmp_path = os.path.join(data_path, 'tmp_{}'.format(self.tmp_time))
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        # Parallelize agent training
        save_model_avg_score = Parallel(n_jobs=32)(
            delayed(train_one_agent)(
                i, self.cfg, self.actor_training_layers, self.critic_training_layers, tmp_path
            ) for i in range(self.num_agents)
        )
        #save_model_avg_score = []
        #for i in range(self.num_agents):
        #    save_model_avg_score.append(train_one_agent(i, self.cfg, self.actor_training_layers, self.critic_training_layers, tmp_path))


        # Aggregate parameters
        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffer = torch.load(file)
            param = []
            for key in buffer.keys():
                if key in self.actor_training_layers or key in self.critic_training_layers:
                    param.append(buffer[key].data.reshape(-1))
            pdata.append(torch.cat(param, 0))

        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'train_layer': self.actor_training_layers + self.critic_training_layers,
            'performance': save_model_avg_score,
            'cfg': config_to_dict(self.cfg)
        }
        torch.save(state_dic, os.path.join(final_path, "data.pt"))

        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_avg_score
        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

