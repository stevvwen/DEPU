
import shutil
from .base_task import BaseTask
from core.utils import *
import glob
import json



import datetime
import wandb
from core.data.agent_trainer import AgentTrainer
from core.runner.runner import *

import math

# tmp
from core.utils.utils import *




class RLTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(RLTask, self).__init__(config, **kwargs)

        self.tmp_time= datetime.datetime.now().strftime('%y%m%d_%H%M%S')

        self.trainer= AgentTrainer(config)

        self.agent_config= config.agent.agent
        self.cfg= config

        self.num_agents= config.num_agents

        self.visualization = not config.debug_mode and config.eval_mode
        self.plot(self.agent_config)

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
        turns= 5

        test_agent, test_env = self.trainer.set_up_test(config= self.cfg)

        test_agent= partial_reverse_tomodel(param, test_agent, self.cfg.train_layer).to(param.device)

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

        return avg_score, avg_step

    def val_g_model(self, input):
        #TODO:
        pass

    # override the abstract method in base_task.py, you obtain the model data for generation
    def train_for_data(self):


        if self.cfg.train_layer == 'all':
            actor_train_layer = [name for name, module in self.trainer.agent.actor.named_parameters()]
            critic_train_layer = [name for name, module in self.trainer.agent.critic.named_parameters()]
        else:
            actor_train_layer = self.agent_config.train_layer
            critic_train_layer = self.agent_config.train_layer

        data_path = getattr(self.cfg, 'save_root', 'param_data')

        tmp_path = os.path.join(data_path, 'tmp_{}'.format(self.tmp_time))
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        save_model_avg_score = []
        highest_avg_score= -math.inf


        for i in range(0, self.num_agents):
            self.trainer.rollout()
            avg_score, eval_epi_steps= self.trainer.evaluate()
            highest_avg_score = max(avg_score, highest_avg_score)

            save_model_avg_score.append(avg_score)
            torch.save(extract_agent_params(actor_train_layer, critic_train_layer, self.trainer.agent),
                       os.path.join(tmp_path, "p_data_{}.pt".format(i)))

            print(f"Agent {i} training complete")

        print("Training complete")

        train_layer= {}
        train_layer.update(actor_train_layer)
        train_layer.update(critic_train_layer)

        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'performance': save_model_avg_score,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_avg_score
        }

        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        #shutil.copy(os.path.abspath(__file__), os.path.join(final_path, os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}
