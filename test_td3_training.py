from datetime import datetime
import wandb
from core.data.agent_trainer import AgentTrainer
from core.runner.runner import *
# tmp
from core.utils.utils import make_env, make_agent


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_td3_for_data(config: DictConfig):

    env, obs_dim, act_dim = make_env(config.rl_env)
    agent= make_agent(obs_dim, act_dim, config.agent.agent)
    replay_buffer= hydra.utils.instantiate(config.replay_buffer)
    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="DEPU",
        name= datetime.now().strftime('%y%m%d_%H%M%S'),
        config=config.agent.agent,
        allow_val_change=True
    )
    trainer= AgentTrainer(config, env= env, model= agent, replay_buffer= replay_buffer)
    trainer.rollout()

    return


if __name__ == "__main__":
    training_td3_for_data()