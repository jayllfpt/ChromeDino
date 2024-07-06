# Check Environment    
import cv2
import matplotlib.pyplot as plt
from environment import ChromeDino


if __name__=="__main__":
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    env = ChromeDino()
    obs = env.get_observation()
    plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_GRAY2BGR))
    plt.imshow(obs[0])
    plt.show()

    