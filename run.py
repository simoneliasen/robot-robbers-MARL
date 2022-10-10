from enum import auto
import os
import gym
from gym import spaces
import math
import matplotlib as plt
import numpy as np
from stable_baselines3 import PPO
from game.environment import RobotRobbersEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import argparse
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback

# Settings argparse
##################################################################################################

parser = argparse.ArgumentParser(description='RoboCopper ArgParse')
parser.add_argument('--mode', type=str,
                    help="enter the model you want to train, either 'PPO', 'A2C' or 'SAC'", default="tune")
parser.add_argument('--load', type=bool,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=False)
parser.add_argument('--multi', type=bool,
                    help="MultiCPU usage or not'", default=False)
parser.add_argument('--cpu', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=4)
parser.add_argument('--maxticktotal', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=100000)
parser.add_argument('--maxtickepisode', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=200)
parser.add_argument('--agentcount', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=1)
parser.add_argument('--rewardshaping', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=True)
parser.add_argument('--hyperparametertuning', type=int,
                    help="do you want to load a previous model? 'TRUE' or 'FALSE'", default=False)

args = parser.parse_args()

ticks_total = args.maxticktotal
ticks_episode = args.maxtickepisode
load_model = args.load
count_cpu = args.cpu
mode = args.mode
multi = args.multi
agentcount = args.agentcount
reward_shaping = args.rewardshaping
hyperparameter_tuning = args.hyperparametertuning

# Save best performing model
##################################################################################################


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} ".format(
                        self.best_mean_reward, mean_reward))
                    if mode == "tune" and hyperparameter_tuning:
                        wandb.log({
                            'mean_reward': mean_reward
                        })

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    # if self.verbose > 0:
                    # print("Saving new best model to {}".format(self.save_path))
                    # self.model.save(self.save_path)
        return True

# Resets Environments after ticks_episode
##################################################################################################


class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=10000):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)

        # Overwrite the done signal when
        if self.current_step >= self.max_steps:
            done = True
        # Update the info dict to signal that the limit was exceeded
        info['time_limit_reached'] = True
        info['Current_Step'] = self.current_step
        return obs, reward, done, info

# Discretize actions. i.e [0.6212321, -0.2333, -0,5232] to [1,0,-1]
##################################################################################################


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        discrete_actions = []
        for value in action:
            rounded = np.round(value).astype(int)
            discrete_actions.append(rounded)
        return discrete_actions

# Reward = inverse distance to nearest cashbag + reaching it bonus
###################################################################################################


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, envregulator, diviser, reachreward):
        super().__init__(env)
        self.envregulator = envregulator
        self.diviser = diviser
        self.reachreward = reachreward

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        cashbags = [(x, y, w, h)
                    for (x, y, w, h) in state[2] if x >= 0 and y >= 0]
        robots = [(x, y, w, h)
                  for (x, y, w, h) in state[0] if x >= 0 and y >= 0]
        indexed_robot_gold_carry = state[5]

        def theSum(aList):
            s = 0
            for x in aList:
                if x > 0:
                    s = s + x
            return s
        indexed_robot_gold_carry = theSum(state[5])
       # print(indexed_robot_gold_carry)

        distances = []
        robotnum = 1
        total_reward = 0

        for robot in robots:
            robot = [robot[0], robot[1]]
            # print("Robot " + str(robotnum) + " " + str(robot))
            robotnum += 1
            cashnum = 1
            distances = []

            for cash in cashbags:
                cash = (cash[0], cash[1])
                distance = -abs(math.dist(robot, cash))
                distances.append(distance)
                # print("Cashbag " + str(cashnum) + " " + str(cash))
                # print("Distance: " + str(distance))
                cashnum += 1
            # rounded = np.round(value).astype(int)
            if len(distances) > 0:

                shortest_dist = max(distances)

                reward += np.round((shortest_dist +
                                   self.envregulator) / self.diviser).astype(int)

                if shortest_dist == 0:
                    reward += self.reachreward

            total_reward += reward

            if mode == "tune" and reward_shaping:
                wandb.log({
                    'totalcashbags': indexed_robot_gold_carry
                })

        return state, total_reward, done, info

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

# Function to create multi-cpu environments
###################################################################################################


def make_env(rank, seed=0, max_ticks_per_episode=2000):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RobotRobbersEnv()
        env.n_robbers = agentcount
        env = ActionWrapper(env)
        env = TimeLimitWrapper(env, max_steps=ticks_episode)
        env = RewardWrapper(env, envregulator, diviser, reachreward)

        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

# Train model
###################################################################################################


def train(ppo_parameters, reward_parameters):

    print("-------- Creating Environment ------------------")

    learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip, entropy_coefficient, value_coefficient = ppo_parameters
    envregulator, diviser, reachreward = reward_parameters

    if multi:
        env = VecMonitor(SubprocVecEnv(
            [make_env(i, ticks_episode) for i in range(args.cpu)]), "tmp/TestMonitor")
    else:
        env = RobotRobbersEnv()
        env.n_robbers = agentcount
        env = ActionWrapper(env)
        env = TimeLimitWrapper(env, max_steps=ticks_episode)
        env = RewardWrapper(env, envregulator, diviser, reachreward)

    env.reset()

    print("-------- Done Creating Environment ------------------")

    print("-------- Starting Learning ------------------")

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=log_dir)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=".\\board",
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip,
                ent_coef=entropy_coefficient,
                vf_coef=value_coefficient,)

    if load_model:
        model = model.load("tmp/best_model", env)

    if mode == "tune":
        model.learn(total_timesteps=config['total_timesteps'],
                    callback=[WandbCallback(
                        gradient_save_freq=100, verbose=2,), callback],
                    tb_log_name="TUNE-SIMPLE-PPO")
    else:
        model.learn(total_timesteps=ticks_total,
                    callback=callback, tb_log_name="TRAIN-SIMPLE-PPO")
        model.save("PPO")

    print("-------- Done  Learning ------------------")

# Tune model
##################################################################################################


def tune(ppo_parameters, reward_parameters, reward_config, hyper_config):

    learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip, entropy_coefficient, value_coefficient = ppo_parameters
    envregulator, diviser, reachreward = reward_parameters

    print("-------- Tuning a model ------------------")  # Might not work as,

    if hyperparameter_tuning:
        wandb.init(config=hyper_config)
        learning_rate = wandb.config['learning_rate']
        batch_size = wandb.config['batch_size']
        n_epochs = wandb.config['n_epochs']
        gamma = wandb.config['gamma']
        gae_lambda = wandb.config['gae_lambda']
        clip = wandb.config['clip']
        entropy_coefficient = wandb.config['entropy_coefficient']
        value_coefficient = wandb.config['value_coefficient']

    if reward_shaping:
        wandb.init(config=reward_config)
        envregulator = wandb.config['envregulator']
        diviser = wandb.config['diviser']
        reachreward = wandb.config['reachreward']

    ppo_parameters = learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip, entropy_coefficient, value_coefficient
    reward_parameters = envregulator, diviser, reachreward

    train(ppo_parameters, reward_parameters)

    print("-------- Tune Done on model ------------------")

# Test model
##################################################################################################


def test():
    print("-------- Inference on model ------------------")

    episodes = 5  # Amount of environments to use inference on
    max_ticks = 350  # max_ticks pr. episode
    tick_count = 0  # Tick counter

   # env = VecMonitor(SubprocVecEnv(make_env("id", ticks_episode)),"tmp/TestMonitor")

    env = RobotRobbersEnv()
    env = ActionWrapper(env)
    env = RewardWrapper(env, envregulator, diviser, reachreward)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./board")

    if load_model:
        model = model.load("tmp/best_model", env)  # loads latest best model

    for episode in range(episodes):
        obs = env.reset()

        for ticks in range(max_ticks):

            action, state = model.predict(obs)  # deterministic=True ?
            obs, reward, done, info = env.step(action)
            env.render()

            tick_count += 1
            # print(info)

    env.close()

    print("-------- Inference Done ------------------")

# Initialize file and parameters
##################################################################################################


if __name__ == '__main__':

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Standard config
    config = {
        "total_timesteps": 300000,
        "env_name": "Robo-Rob-1",
    }

    # Hyper tuning config
    hyper_config = {
        'method': 'bayes',  # grid, random, bayesian
        'metric': {
            'name': 'totalcashbags',  # mean_reward
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'values': [64, 128, 256]
            },
            'n_epochs': {
                'values': [3, 5, 10]  # reduced
            },
            'gamma': {
                'values': [0.95, 0.97, 0.99]
            },
            'gae_lambda': {
                'values': [0.95]
            },
            'learning_rate': {
                'values': [0.000005, 0.00005, 0.0005]
            },
            'clip': {
                'values': [0.1, 0.2]  # reduced
            },
            'value_coefficient': {
                'values': [0.5, 1]
            },
            'entropy_coefficient': {
                'values': [0, 0.01]
            }
        }
    }

    # Reward tuning config
    reward_config = {
        'method': 'bayes',  # grid, random, bayesian
        'metric': {
            'name': 'totalcashbags',  # mean_reward
            'goal': 'maximize'
        },
        'parameters': {
            'envregulator': {
                'values': [0, 2, 5, 10, 20, 30, 40, 50, 60, 80, 90, 128]
            },
            'diviser': {
                'values': [2, 3, 4, 5, 10, 15, 20, 25, 35, 40, 50, 75, 100]
            },
            'reachreward': {
                'values': [3, 6, 9, 15, 25, 40, 50, 70, 100, 200, 500, 600, 900, 1000, 5000]
            }
        }

    }

    # Default PPO parameters
    learning_rate = 0.000005
    batch_size = 64
    n_epochs = 3
    gamma = 0.99
    gae_lambda = 0.95
    clip = 0.1
    entropy_coefficient = 0.01
    value_coefficient = 0.5
    ppo_parameters = learning_rate, batch_size, n_epochs, gamma, gae_lambda, clip, entropy_coefficient, value_coefficient

    # Default Reward shaping parameters
    envregulator = 40
    diviser = 2
    reachreward = 700
    reward_parameters = envregulator, diviser, reachreward

    if mode == "train":
        train(ppo_parameters, reward_parameters)
    elif mode == "test":
        test()
    elif mode == "tune":
        if reward_shaping:
            sweep_id = wandb.sweep(sweep=reward_config, project="sb3")
        if hyperparameter_tuning:
            sweep_id = wandb.sweep(sweep=hyper_config, project="sb3")
        count = 100
        print("\n>>Running Sweep " + str(count) + " times")
        wandb.agent(sweep_id=sweep_id, function=tune(
            ppo_parameters, reward_parameters, reward_config, hyper_config), count=count)

##################################################################################################
