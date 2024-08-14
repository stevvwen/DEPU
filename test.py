from core.runner.runner import *
# tmp
from core.utils.utils import make_env, make_agent


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_agent(config: DictConfig):

    env, eval_env, env_dict = make_env(config.rl_env)
    agent= make_agent(env_dict, config.agent.agent)

    train_layer = [name for name, module in agent.actor.named_parameters()]
    print(train_layer)
    
    train_layer = [name for name, module in agent.critic.named_parameters()]
    print(train_layer)
    
    
    

if __name__ == "__main__":
    training_agent()