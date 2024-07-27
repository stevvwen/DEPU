import torch
import wandb


class AgentTrainer:
    def __init__(self, config, env, eval_env, model, replay_buffer):
        self.env = env
        self.eval_env= eval_env
        self.reward_trace = 0  # trace of reward

        self.batch_size = config.batch_size
        self.updates = 0
        self.cur_state, _ = env.reset()
        self.epi_reward = 0
        self.epi_count = 0
        self.max_steps = config.max_steps
        self.debug_mode = config.debug_mode
        self.buffer = replay_buffer
        self.model= model
        self.eval_interval= config.eval_interval
        self.eval_mode= False

        self.cum_reward= 0

        self.avg_epi_reward= 0

        self.eval_count= 0

    def rollout(self):


        print(f"Agent training starts")
        state = self.cur_state

        for step in range(self.max_steps):

            action = self.model.act(state, step, False)  # Select action
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Step

            self.epi_reward += reward
            self.cum_reward += reward


            reward= reward/ 16 #TODO: Change this to a more general form

            # If the episode is truncated, we still do the update
            mask = 1 if truncated else float(not terminated)

            self.buffer.push(state, action, reward, next_state, mask)  # Append transition to memory

            if not self.debug_mode:
                wandb.log({"Avg Reward": self.cum_reward/(step+ 1), "custom_step": step})

            state = next_state

            if terminated or truncated:

                if not self.debug_mode:
                    #wandb.log({"Episode": self.epi_count, "Epi Reward": self.epi_reward})
                    wandb.log({"Epi Reward": self.epi_reward, "epi_count": self.epi_count})
                self.epi_count += 1
                self.epi_reward = 0
                state, _ = self.env.reset()

            if step % self.eval_interval == 0:
                
                self.evaluate()
                

                

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

        self.eval_mode = True
        total_score= 0
        eval_epi_steps = 0
        for _ in range(turns):
            state, _ = self.eval_env.reset()

            done= False

            while not done:
                
                action = self.model.act(state, self.model.num_expl_steps, True)  # Select action

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)  # Step
                done= terminated or truncated
                total_score+= reward

                eval_epi_steps+= 1
                state= next_state

        avg_score= total_score/turns
        avg_step= eval_epi_steps/turns

        self.eval_count+= 1

        if not self.debug_mode:
            wandb.log({"Eval Epi Reward": avg_score, "eval_epi_count": self.eval_count})

        print(f"Agent evaluation turn {self.eval_count} "
              f"with Episode Reward {avg_score} and Episode Length {avg_step}")

        self.eval_mode = False
        return avg_score, eval_epi_steps
