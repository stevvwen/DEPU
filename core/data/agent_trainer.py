import torch
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
        self.eval_eval_interval= config.eval_eval_interval
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

            reward= (reward+ 8)/ 8 #TODO: Change this to a more general form

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

            if step % self.eval_eval_interval == 0:
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


    def evaluate(self, turns= 3):


        total_score= 0
        eval_epi_steps = 0
        for _ in range(turns):
            state, _ = self.env.reset()

            done= False

            while not done:
                with torch.no_grad():
                    action = self.model.act(state, self.model.num_expl_steps, False)  # Select action

                next_state, reward, terminated, truncated, info = self.env.step(action)  # Step
                done= terminated or truncated
                total_score+= reward

                eval_epi_steps+= 1
                state= next_state

        avg_score= total_score/turns
        avg_step= eval_epi_steps/turns

        self.eval_count+= 1

        if not self.debug_mode:
            wandb.log({"Eval Reward": avg_score, "eval_epi_count": self.eval_count})

        print(f"Agent evaluation turn {self.eval_count} "
              f"with Episode Reward {avg_score} and Episode Length {avg_step}")

        return avg_score, eval_epi_steps
