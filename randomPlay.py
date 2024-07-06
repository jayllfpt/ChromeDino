import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3 import DQN
from environment.ChromeDino import * 

if __name__=="__main__":   
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'  
 
    env = ChromeDino()

    for episode in range(10): 
        obs = env.reset()
        done = False  
        total_reward = 0
        while not done: 
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward  += reward 
        print('Total Reward for episode {} is {}'.format(episode, total_reward))  