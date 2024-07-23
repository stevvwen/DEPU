import datetime
import wandb
from core.data.agent_trainer import AgentTrainer
from core.runner.runner import *
# tmp
from core.utils.utils import make_env, make_agent


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_td3_for_data(config: DictConfig):

    env, env_dict = make_env(config.rl_env)
    agent= make_agent(env_dict, config.agent.agent)
    replay_buffer= hydra.utils.instantiate(config.replay_buffer)
    
    tmp_name= datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    agent_cfg = config.agent.agent
    if not config.debug_mode:
        wandb.login()


        run = wandb.init(
            # Set the project where this run will be logged
            project="DEPU",
            name= tmp_name,
            config={"learning rate": agent_cfg.lr,
                    "observation size": agent_cfg.obs_shape,
                    "action size": agent_cfg.act_shape,
                    "hidden size": agent_cfg.hidden_dim,
                    },
            allow_val_change=True
        )

        # define our custom x axis metric
        wandb.define_metric("custom_step", hidden=True)
        wandb.define_metric("epi_count", hidden= True)
        wandb.define_metric("eval_epi_count", hidden= True)

        # define which metrics will be plotted against it
        wandb.define_metric("Avg Reward", step_metric="custom_step")
        wandb.define_metric("Epi Reward", step_metric="epi_count")
        wandb.define_metric("Eval Epi Reward", step_metric="eval_epi_count")


    trainer= AgentTrainer(config, env= env, model= agent, replay_buffer= replay_buffer)
    trainer.rollout()

    return


if __name__ == "__main__":
    training_td3_for_data()