import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker 
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import *


if __name__=="__main__":
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    env = WebGame()

    model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)

    model.load('train/best_model_53000.zip') 

    for episode in range(5): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(int(action))
            time.sleep(0.01)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode, total_reward))
        time.sleep(2)