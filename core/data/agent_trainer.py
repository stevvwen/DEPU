import wandb


class AgentTrainer:
    def __init__(self, config, env, model, replay_buffer):
        self.env = env
        self.reward_trace = 0  # trace of reward

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
        self.eval_epi_freq= config.eval_episode_freq
        self.eval_mode= False

        self.cum_reward= 0

        self.avg_epi_reward= 0

        self.eval_count= 0

    def rollout(self):


        print(f"Agent training starts")
        state = self.cur_state

        # Rollout a trajectory with 16 samples
        for step in range(self.max_steps):

            action = self.model.act(state, step, False)  # Select action
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Step
            self.epi_steps += 1

            # reward= 16*reward # normalize reward
            # If the episode is truncated, we still do the update
            mask = 1 if truncated else float(not terminated)

            self.buffer.push(state, action, reward, next_state, mask)  # Append transition to memory

            self.epi_reward += reward

            self.cum_reward+= reward

            wandb.log({"Avg Reward": self.cum_reward/(step+ 1), "custom_step": step})

            state = next_state

            if terminated or truncated:

                if not self.debug_mode:
                    #wandb.log({"Episode": self.epi_count, "Epi Reward": self.epi_reward})
                    wandb.log({"Epi Reward": self.epi_reward, "epi_count": self.epi_count})
                # print("Here", self.epi_reward, self.epi_steps, self.total_steps)
                self.epi_count += 1
                self.epi_reward = 0
                self.epi_steps = 0

                if self.epi_count % self.eval_epi_freq == 0:
                    self.eval_mode = True
                    self.evaluate()
                    self.eval_mode = False

                state, _ = self.env.reset()

            self.cur_state = state

            if len(self.buffer) < self.batch_size:
                continue
            else:
                batch = self.buffer.sample(
                    self.batch_size)
                self.model.update(batch, step)

        print("Single agent training complete")
        return


    def evaluate(self):
        print(f"Agent evaluation episode {self.epi_count/self.eval_epi_freq} starts")

        state, _ = self.env.reset()

        terminated= False
        truncated= False

        eval_rewards= 0
        eval_epi_steps= 0

        while not (terminated or truncated):
            action = self.model.act(state, self.model.num_expl_steps, self.eval_mode)  # Select action
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Step

            eval_rewards+= reward

            eval_epi_steps+= 1

        if not self.debug_mode:
            wandb.log({"Eval Epi Reward": eval_rewards, "eval_epi_count": self.eval_epi_freq})

        print(f"Agent evaluation episode {self.epi_count/self.eval_epi_freq} "
              f"with Episode Reward {eval_rewards} and Episode Length {eval_epi_steps}")

        return eval_rewards, eval_epi_steps

