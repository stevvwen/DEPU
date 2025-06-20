import gymnasium as gym
import hydra
import numpy as np
import torch
import omegaconf
import os
import random

def get_weights(train_list, net):
    # Extract weights from the network based on the training list
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param

def extract_agent_params(actor_training_layers, critic_training_layers, agent):
    # Extract the parameters of the agent's actor and critic networks
    params= {}
    params.update(get_weights(actor_training_layers, agent.actor))
    params.update(get_weights(critic_training_layers, agent.critic))
    return params


def set_flat_params(flattened, model, train_layer):
    # Set the parameters of the model with the flattened parameters
    layer_idx = 0
    for name, pa in model.named_parameters():
        if name in train_layer:
            pa_shape = pa.shape
            pa_length = pa.view(-1).shape[0]
            pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
            pa.data.to(flattened.device)
            layer_idx += pa_length
    return model, layer_idx

def replace_agent(flattened, model, actor_train_layer, critic_train_layer, actor_num, critic_num):
    # Replace the parameters of the agent's actor and critic networks with the flattened parameters
    model.actor, actor_num_param = set_flat_params(flattened[:actor_num], model.actor, actor_train_layer)
    model.critic, critic_num_param = set_flat_params(flattened[actor_num:actor_num + critic_num], model.critic, critic_train_layer)
    assert actor_num_param == actor_num
    assert critic_num_param == critic_num
    return model


def replace_policy(flattened, model, actor_train_layer, actor_num):
    model.actor, actor_num_param = set_flat_params(flattened[:actor_num], model.actor, actor_train_layer)
    assert actor_num_param == actor_num
    return model



def get_storage_usage(path):
    list1 = []
    fileList = os.listdir(path)
    for filename in fileList:
        pathTmp = os.path.join(path,filename)  
        if os.path.isdir(pathTmp):   
            get_storage_usage(pathTmp)
        elif os.path.isfile(pathTmp):  
            filesize = os.path.getsize(pathTmp)  
            list1.append(filesize) 
    usage_gb = sum(list1)/1024/1024/1024
    return usage_gb


def test_g_model(self, trainer, param, turns=20):
        """
        Evaluate a single policy parameter vector over `turns` episodes.
        """
        test_agent, test_env = trainer.setup_test()
        param_tensor = torch.squeeze(param).to(test_agent.device)
        test_agent = replace_policy(
            param_tensor, test_agent, self.actor_layers,
            actor_num=sum(
                torch.numel(m)
                for n, m in test_agent.actor.named_parameters()
                if n in self.actor_layers
            )
        )

        total_reward, total_steps = 0.0, 0
        for _ in range(turns):
            state, _ = test_env.reset()
            done = False
            while not done:
                action, = test_agent.act(state, test_agent.num_expl_steps, True),
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_steps += 1
                state = next_state

        avg_reward = total_reward / turns
        avg_steps  = total_steps / turns
        return avg_reward, avg_steps



def config_to_dict(config):
    return omegaconf.OmegaConf.to_container(config, resolve=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    torch.cuda.set_device(device_config)
    torch.set_float32_matmul_precision('high')
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg):

    # Initialize the experiment with the given configuration
    set_seed(cfg.seed)
    set_device(cfg.device)
    set_processtitle(cfg)
    
    # Create directories for saving results
    save_root = getattr(cfg, "save_root", "results")
    os.makedirs(save_root, exist_ok=True)
    
    # Log the configuration
    config_dict = config_to_dict(cfg)
    #print("Experiment Configuration:", config_dict)

    return save_root, config_dict


