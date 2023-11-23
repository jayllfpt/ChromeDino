import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker 
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import *
import matplotlib.pyplot as plt


if __name__=="__main__":
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    env = WebGame()
    # env_checker.check_env(env)
    obs=env.get_observation()
    plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_GRAY2BGR))
    plt.imshow(obs[0])
    
    plt.show()