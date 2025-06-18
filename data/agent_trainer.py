import torch
import hydra
import wandb
from utils import make_env, make_agent

class AgentTrainer:
    def __init__(self, config):
        self.config       = config
        self.device       = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size   = config.batch_size
        self.max_steps    = config.rl_env.max_steps
        self.debug_mode   = config.debug_mode
        self.eval_interval= config.eval_interval
        self.eval_mode    = config.eval_mode

        self._setup_task()
        self.reset_trainer()

    def _setup_task(self):
        self.env, self.eval_env, self.env_dict = make_env(self.config.rl_env)
        self.config.agent.agent.device = self.device
        self.agent  = make_agent(self.env_dict, self.config.agent.agent)
        self.buffer = hydra.utils.instantiate(self.config.replay_buffer)

    def reset_trainer(self):
        # reset counters and data
        self.epi_reward  = 0
        self.epi_count   = 0
        self.cum_reward  = 0
        self.eval_count  = 0

        # re-instantiate agent & buffer for a fresh start
        self.config.agent.agent.device = self.device
        self.agent  = make_agent(self.env_dict, self.config.agent.agent)
        self.buffer = hydra.utils.instantiate(self.config.replay_buffer)

        self.cur_state, _ = self.env.reset()

    def rollout(self):
        state = self.cur_state
        for step in range(self.max_steps):
            action = self.agent.act(state, step, False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.epi_reward += reward
            self.cum_reward += reward
            norm_reward    = reward / 8  # TODO: generalize this

            mask = 1 if truncated else float(not terminated)
            self.buffer.push(state, action, norm_reward, next_state, mask)

            if self.eval_mode and not self.debug_mode:
                wandb.log({
                    "Avg Reward": self.cum_reward / (step + 1),
                    "custom_step": step
                })

            state = next_state

            if terminated or truncated:
                if self.eval_mode and not self.debug_mode:
                    wandb.log({
                        "Epi Reward": self.epi_reward,
                        "epi_count": self.epi_count
                    })
                self.epi_count += 1
                self.epi_reward = 0
                state, _ = self.env.reset()

            if step % self.eval_interval == 0 and self.eval_mode:
                self.evaluate()

            self.cur_state = state

            # only update once we have enough samples and past exploration
            if len(self.buffer) >= self.batch_size and step >= self.agent.num_expl_steps:
                batch = self.buffer.sample(self.batch_size)
                self.agent.update(batch, step)

    def setup_test(self):
        self.eval_env.reset()
        return make_agent(self.env_dict, self.config.agent.agent), self.eval_env

    def evaluate(self, turns=3):
        total_score, total_steps = 0, 0
        for _ in range(turns):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.agent.act(state, self.agent.num_expl_steps, True)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_score += reward
                total_steps += 1
                state = next_state

        avg_score = total_score / turns
        avg_step  = total_steps  / turns
        self.eval_count += 1

        if self.eval_mode and not self.debug_mode:
            wandb.log({
                "Eval Epi Reward": avg_score,
                "eval_epi_count": self.eval_count
            })

        return avg_score, avg_step
