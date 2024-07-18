from agents.td3 import *
import hydra
from omegaconf import DictConfig
from core.runner.runner import *
from core.data.agent_trainer import AgentTrainer

#tmp
from core.utils.utils import make_env, make_agent

@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_td3_for_data(config: DictConfig):

    env, obs_dim, act_dim = make_env(config.rl_env)
    agent= make_agent(obs_dim, act_dim, config.agent.agent)
    replay_buffer= hydra.utils.instantiate(config.replay_buffer)
    trainer= AgentTrainer(config, env= env, model= agent, replay_buffer= replay_buffer)
    trainer.rollout()

    return


if __name__ == "__main__":
    training_td3_for_data()