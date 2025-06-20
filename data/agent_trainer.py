import torch
import hydra
import gymnasium as gym

class AgentTrainer:
    def __init__(self, config):
        self.config       = config
        self.device       = torch.device(config.device)
        self.batch_size   = config.batch_size
        self.max_steps    = config.rl_env.max_steps
        self.eval_interval= config.eval_interval
        self.eval_mode    = config.eval_mode
        self.reset()
        

    def reset(self):
        """Reset trainer state and create fresh agent/buffer"""
        self.agent = hydra.utils.instantiate(self.config.agent.agent)
        self.buffer = hydra.utils.instantiate(self.config.replay_buffer)
        self.env= gym.make(self.config.rl_env.env_name, **self.config.rl_env.env_kwargs)
        self.eval_env = gym.make(self.config.rl_env.env_name, **self.config.rl_env.env_kwargs)
        self.state, _ = self.env.reset()
        
        # Reset counters
        self.episode_reward = 0
        self.episode_count = 0

    def rollout(self):
        """Main training loop"""
        for step in range(self.max_steps):
            self._step(step)
            
            if step % self.eval_interval == 0 and self.eval_mode:
                self.evaluate()

    def _step(self, step):
        """Single environment step"""
        action = self.agent.act(self.state, step, False)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        # Update rewards and buffer
        self.episode_reward += reward
        norm_reward = reward / 8  # TODO: make configurable
        mask = 1 if truncated else float(not terminated)
        self.buffer.push(self.state, action, norm_reward, next_state, mask)
        
        # Handle episode end
        if terminated or truncated:
            self.episode_count += 1
            self.episode_reward = 0
            self.state, _ = self.env.reset()
        else:
            self.state = next_state
            
        # Update agent if ready
        if len(self.buffer) >= self.batch_size and step >= self.agent.num_expl_steps:
            batch = self.buffer.sample(self.batch_size)
            self.agent.update(batch, step)

    @torch.no_grad()
    def evaluate(self, num_episodes=3):
        """Evaluate agent performance"""
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            while True:
                action = self.agent.act(state, self.agent.num_expl_steps, True)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        return avg_reward

    def setup_test(self):
        """Setup for testing"""
        self.eval_env.reset()
        return make_agent(self.env_dict, self.config.agent.agent), self.eval_env