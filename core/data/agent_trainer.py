import wandb
from core.utils.utils import *

class AgentTrainer:
    def __init__(self, config):

        self.batch_size = config.batch_size
        self.max_steps = config.rl_env.max_steps
        self.debug_mode = config.debug_mode
        self.eval_interval= config.eval_interval
        self.eval_mode= config.eval_mode

        self.device= "cuda:0" if torch.cuda.is_available() else "cpu"

        self.config= config

        self.setup_task(config)
        self.reset_trainer()

    def setup_task(self, config):
        self.env, self.eval_env, self.env_dict = make_env(config.rl_env)
        config.agent.agent.device = self.device
        self.agent = make_agent(self.env_dict, config.agent.agent)
        self.buffer = hydra.utils.instantiate(config.replay_buffer)

    def reset_trainer(self):

        config= self.config

        self.updates = 0
        self.epi_reward = 0
        self.epi_count = 0
        self.cum_reward = 0
        self.avg_epi_reward = 0
        self.eval_count = 0

        config.agent.agent.device = self.device
        self.agent= make_agent(self.env_dict, config.agent.agent)
        self.buffer = hydra.utils.instantiate(config.replay_buffer)
        self.cur_state, _ = self.env.reset()

    def set_up_test(self, config):
        self.eval_env.reset()
        return make_agent(self.env_dict, config.agent.agent), self.eval_env


    def rollout(self):

        #print(f"Agent training starts")
        state = self.cur_state

        for step in range(self.max_steps):
            action = self.agent.act(state, step, False)  # Select action
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Step

            self.epi_reward += reward
            self.cum_reward += reward


            reward= reward/ 8 #TODO: Change this to a more general form

            # If the episode is truncated, we still do the update
            mask = 1 if truncated else float(not terminated)

            self.buffer.push(state, action, reward, next_state, mask)  # Append transition to memory

            if not self.debug_mode and self.eval_mode:
                wandb.log({"Avg Reward": self.cum_reward/(step+ 1), "custom_step": step})

            state = next_state

            if terminated or truncated:

                if not self.debug_mode and self.eval_mode:
                    wandb.log({"Epi Reward": self.epi_reward, "epi_count": self.epi_count})
                self.epi_count += 1
                self.epi_reward = 0
                state, _ = self.env.reset()

            if step % self.eval_interval == 0 and self.eval_mode:
                
                self.evaluate()


            self.cur_state = state

            if len(self.buffer) < self.batch_size or step< self.agent.num_expl_steps:
                continue
            else:
                batch = self.buffer.sample(
                    self.batch_size)
                self.agent.update(batch, step)
        return


    def evaluate(self, turns= 3):


        total_score= 0
        eval_epi_steps = 0
        for _ in range(turns):
            state, _ = self.eval_env.reset()

            done= False

            while not done:
                
                action = self.agent.act(state, self.agent.num_expl_steps, True)  # Select action

                next_state, reward, terminated, truncated, info = self.eval_env.step(action)  # Step
                done= terminated or truncated
                total_score+= reward

                eval_epi_steps+= 1
                state= next_state

        avg_score= total_score/turns
        avg_step= eval_epi_steps/turns

        self.eval_count+= 1

        if not self.debug_mode and self.eval_mode:
            wandb.log({"Eval Epi Reward": avg_score, "eval_epi_count": self.eval_count})


        return avg_score, avg_step
