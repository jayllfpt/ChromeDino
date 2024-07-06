from mss import mss
import pydirectinput
import cv2
import numpy as np
import easyocr
from gym import Env
from gym.spaces import Box, Discrete
import time

class ChromeDino(Env):
    def __init__(self):
        super().__init__()
        self.shape0 = 83
        self.shape1 = 100
        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,self.shape0,self.shape1), dtype=np.uint8)
        self.action_map = {
            0: 'no_act',
            1: 'space',
            # 2: 'down', 
        }
        self.action_space = Discrete(int(len(self.action_map.keys())))

        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 230, 'left': 230, 'width': 120, 'height': 55}
        self.done_location = {'top': 190, 'left': 360, 'width': 90, 'height': 30}
        self.ocr = easyocr.Reader(lang_list=['en'], gpu = True, detector=False)
        
    def step(self, action):
        if action !=0:
        # if action == 0:
            pydirectinput.press(self.action_map[action])
        done = self.get_done() 
        observation = self.get_observation()
        reward = 1
        print('[o] - Action:', self.action_map[action])
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        time.sleep(1)
        print("[x] - Reset")
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()
        
    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
         
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("gameFrame.jpg", raw)
        resized = cv2.resize(gray, (self.shape1,self.shape0))
        channel = np.reshape(resized, (1,self.shape0,self.shape1))
        return channel
    
    def get_done(self):
        done_strings = ['GAME', 'GAHE', '6AME']
        done = False
        try:
            res = self.ocr.recognize(
                np.array(self.cap.grab(self.done_location))
            )[0][1]
            if res.replace(" ", "") in done_strings:
                done = True
        except:
            done = False
        return done