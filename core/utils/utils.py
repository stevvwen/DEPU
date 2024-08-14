import re

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def state_part(train_list, net):
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param

def extract_agent_params(actor_train_layer, critic_train_layer, agent):
    params= {}
    params.update(state_part(actor_train_layer, agent.actor))
    params.update(state_part(critic_train_layer, agent.critic))
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
    return model


def test_generated_partial(net, param, train_layer, dataloader, fea_path=None):
    target_num = 0
    for name, module in net.named_parameters():
        if name in train_layer:
            target_num += torch.numel(module)
    params_num = torch.squeeze(param).shape[0]  # + 30720
    assert (target_num == params_num)
    param = torch.squeeze(param)
    net = partial_reverse_tomodel(param, net, train_layer).to(param.device)
    acc, loss, output_list = test(net, dataloader, fea_path=fea_path)
    del net
    return acc, output_list


def test(model, test_loader,fea_path=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    output_list = []

    with torch.no_grad():
        if fea_path is not None:
            expert_files = os.listdir(fea_path)
            for m in expert_files:
                models = fea_path + m
                fea_targets = torch.load(models)
                targets = fea_targets[1].to('cuda')
                inputs = fea_targets[0].to('cuda')
                outputs = model.forward_norm(inputs)
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            return acc, _
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            output_list += pred.cpu().numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    del model
    return acc, test_loss, output_list


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

def reverse_tomodel(flattened, model):
    example_parameters = [p for p in model.parameters()]
    length = 0
    reversed_params = []

    for p in example_parameters:
        flattened_params = flattened[length: length+p.numel()]
        reversed_params.append(flattened_params.reshape(p.shape))
        length += p.numel()

    layer_idx = 0
    for p in model.parameters():
        p.data = reversed_params[layer_idx]
        p.data.to(flattened.device)
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def test_ensem(self, best_params, net):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    ensemble_model = reverse_tomodel(mean, net)
    acc,_= test(ensemble_model.cuda(), self.testloader)
    del best_params
    return acc

def test_ensem_partial(self, best_params,dataloader,fea_path=None):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    acc = test_generated_partial(self, mean,dataloader,fea_path=fea_path)
    del best_params
    return acc


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



