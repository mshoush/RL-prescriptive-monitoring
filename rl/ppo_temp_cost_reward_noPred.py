import gym

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os

# import envs
from sys import argv

import os
import psutil

# Get the process id of the current process
#pid = os.getpid()
#print(pid)

# Get the Process instance associated with the process ID
#p = psutil.Process(pid)

# Set the processor affinity to the first core (CPU 0)
#p.cpu_affinity([3,4,5,6])

inner_env = None


def run(
    env_string,
    policy="MlpPolicy",
    learning_steps=4300,
    verbose=0,
    n_steps=128,
    nminibatches=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    cliprange=0.2,
    cliprange_vf=None,
    lam=0.95,
    policy_kwargs=None,
    tensorboard_log="tensorboard",
    seed=22
    #results_dir = None
):
    #startTime = datetime.now()
    #print(f"Start time: {startTime}")
    import time
    import datetime


    print("\nHERE\n")
    #startTime = datetime.datetime.now()
    startTimes = time.time() #datetime.datetime.now().second
    #print("Start time: %s", startTime)
    print(f"\nStart time in secs: {startTimes}\n")
    

    global inner_env
    inner_env = gym.make(env_string)
    env = DummyVecEnv([lambda: inner_env])

    model = PPO(
        policy=policy,
        env=env,
        verbose=verbose,
        n_steps=n_steps,
        batch_size=nminibatches,
        gamma=gamma,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_range=cliprange,
        clip_range_vf=cliprange_vf,
        gae_lambda=lam,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        #seed=22
    )
    model.learn(
        total_timesteps=learning_steps,
        tb_log_name=os.path.basename(__file__).rstrip(".py"),
        callback=TensorboardCallback(),
    )  #

    #endTime = datetime().now()
    #print(f"\nEnd Time: {endTime}\n")
    #hours, rem = divmod(endTime-startTime, 3600)
    #minutes, seconds = divmod(rem, 60)
    #print("\nTime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    #endTime = datetime.datetime.now() #datetime.now()
    endTimes = time.time() #datetime.datetime.now().second  #time.time() #datetime.now().total_seconds()
    #print(f"endTime: {endTime}")
    print(f"endTime in secs: {endTimes}")
    self.elapsedTimes = endTimes - startTimes
    print(f"ElapsedTimes: {self.elapsedTimes}")


    env.close()


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self._log_freq = 20  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        """
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        """
        if self.n_calls % self._log_freq == 0:
            rewards = self.locals["rewards"]
            for i in range(self.locals["env"].num_envs):
                self.tb_formatter.writer.add_scalar(
                    "rewards/env #{}".format(i + 1), rewards[i], self.n_calls
                )


def tensorboard_callback(locals_, globals_):
    global inner_env
    locals_["self"]
    if inner_env.summary_writer is None:
        inner_env.summary_writer = locals_["writer"]

    return True


if __name__ == "__main__":
    from datetime import datetime
    import time
    # results_dir = argv[1]
    # print(f"\nResults dir{results_dir}\n")

    run(
        #results_dir=results_dir,
        env_string="envs:state-with-temporal-cost-reward-conformal-newReward-v1",
        #env_string="envs:state-with-temporal-cost-reward-conformal-newReward-v0",
        #env_string="envs:state-with-temporal-cost-reward-withoutpred-v0",
        #state-with-temporal-cost-reward-withoutpred-v0
        learning_steps=500000,
        
    )
    
