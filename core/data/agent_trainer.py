from collections import deque

import wandb


class AgentTrainer:
    def __init__(self, config, env, model, replay_buffer, eval_mode= False):
        self.env = env
        self.reward_trace = 0  # trace of reward
        self.cumulative_rewards = deque(maxlen=101)  # cumulative rewards

        self.average_reward = [0] * 10
        self.batch_size = config.batch_size
        self.updates = 0
        self.cur_state, _ = env.reset()
        self.epi_reward = 0
        self.epi_count = 0
        self.epi_steps = 0
        self.max_steps = config.max_steps
        self.debug_mode = config.debug_mode
        self.buffer = replay_buffer
        self.model= model
        self.eval_mode= eval_mode

    def rollout(self):


        print(f"Agent training starts")
        state = self.cur_state

        # Rollout a trajectory with 16 samples
        for step in range(self.max_steps):



            action = self.model.act(state, step, self.eval_mode)  # Select action
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Step
            self.epi_steps += 1

            # reward= 16*reward # normalize reward
            # If the episode is truncated, we still do the update
            mask = 1 if truncated else float(not terminated)

            self.buffer.push(state, action, reward, next_state, mask)  # Append transition to memory

            self.epi_reward += reward

            state = next_state

            if terminated or truncated:

                if not self.debug_mode:
                    #wandb.log({"Episode": self.epi_count, "Epi Reward": self.epi_reward})
                    wandb.log({"Epi Reward": self.epi_reward})
                # print("Here", self.epi_reward, self.epi_steps, self.total_steps)
                self.epi_count += 1
                self.epi_reward = 0
                self.epi_steps = 0
                state, _ = self.env.reset()

            self.cur_state = state

            if len(self.buffer) < self.batch_size:
                continue
            else:
                batch = self.buffer.sample(
                    self.batch_size)
                self.model.update(batch, step)


            return
