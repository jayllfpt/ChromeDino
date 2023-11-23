import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import *
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Run a DQN model.")
parser.add_argument('--model', type=str, default="models/2actions_100000.zip", help='Path to the model file.')
parser.add_argument('--runtimes', type=int, default=5, help='Number of runs to execute.')
args = parser.parse_args()

model = args.model
runtimes = args.runtimes

if __name__ == "__main__":
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    env = WebGame()

    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)

    model.load(args.model)

    for episode in range(runtimes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(int(action))
            time.sleep(0.01)
            total_reward += reward
        print(f'Total Reward for episode {episode} is {total_reward}')
        time.sleep(2)
