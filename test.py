# Make a TD3 agent

from agents.td3 import TD3Agent
import torch


# Make a TD3 agent

agent = TD3Agent(obs_shape=(3,), act_shape=(1,), act_limit_low=-1, act_limit_high=1, device='cuda:0', lr=0.001, hidden_dim=256,
                    critic_target_tau=0.005, num_expl_steps=2000, update_every_steps=2,
                    stddev_schedule=[0.1, 0.1], stddev_clip=0.5, discount=0.99)

actor_num= 0
for name, module in agent.actor.named_parameters():
    actor_num += torch.numel(module)
    print(name, torch.numel(module))

print(actor_num)

critic_num= 0
for name, module in agent.critic.named_parameters():
    critic_num += torch.numel(module)
    print(name, torch.numel(module))
    
print(critic_num)