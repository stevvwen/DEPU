import re
import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf


def state_part(train_list, net):
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param

def extract_agent_params(actor_training_layers, critic_training_layers, agent):
    params= {}
    params.update(state_part(actor_training_layers, agent.actor))
    params.update(state_part(critic_training_layers, agent.critic))
    return params



def fix_partial_model(train_list, net):
    print(train_list)
    for name, weights in net.named_parameters():
        if name not in train_list:
            weights.requires_grad = False

def partial_reverse_tomodel(flattened, model, train_layer):
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

    model.actor, actor_num_param = partial_reverse_tomodel(flattened[:actor_num], model.actor, actor_train_layer)
    model.critic, critic_num_param = partial_reverse_tomodel(flattened[actor_num:actor_num + critic_num], model.critic, critic_train_layer)

    assert actor_num_param == actor_num
    assert critic_num_param == critic_num

    return model


def replace_policy(flattened, model, actor_train_layer, actor_num):
    model.actor, actor_num_param = partial_reverse_tomodel(flattened[:actor_num], model.actor, actor_train_layer)

    assert actor_num_param == actor_num

    return model



def top_acc_params(self, accs, params, topk):
    sorted_list = sorted(accs, reverse=True)[:topk]
    max_indices = [accs.index(element) for element in sorted_list]
    best_params = params[max_indices, :]
    del params
    return best_params

def _warmup_beta(start, end, n_timestep, warmup_frac):

    betas               = end * torch.ones(n_timestep, dtype=torch.float64)
    warmup_time         = int(n_timestep * warmup_frac)
    betas[:warmup_time] = torch.linspace(start, end, warmup_time, dtype=torch.float64)

    return betas

def make_beta_schedule(schedule, start, end, n_timestep):
    if schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'linear':
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / (torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64))
    else:
        raise NotImplementedError(schedule)

    return betas

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


import os


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




# Utility Code for TD3 agent

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def to_float_tensor(batch):
    return tuple(torch.FloatTensor(x) for x in batch)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def make_agent(env_dict, cfg):
    cfg.obs_shape = env_dict["obs_shape"]
    cfg.act_shape = env_dict["act_shape"]
    cfg.act_limit_low= float(env_dict["act_low"])
    cfg.act_limit_high= float(env_dict["act_high"])
    return hydra.utils.instantiate(cfg)

# Util code for env code

def make_env(cfg):
    env= gym.make(cfg.env_name, **cfg.env_kwargs)
    eval_env= gym.make(cfg.env_name, **cfg.env_kwargs)
    return env, eval_env, {"obs_shape": env.observation_space.shape, "act_shape": env.action_space.shape,
                 "act_high": env.action_space.high[0], "act_low": env.action_space.low[0]}


def config_to_dict(config):
    return omegaconf.OmegaConf.to_container(config, resolve=True)