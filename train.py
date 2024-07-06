import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from model import DQNModel  
from environment.ChromeDino import * 

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq 
        self.save_path = save_path

    def _init_callback(self): 
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0: 
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

  
if __name__=="__main__":   
    CHECKPOINT_DIR = './checkpoints/'
    LOG_DIR = './logs/'  
 
    env = ChromeDino()
      
    callback = TrainAndLoggingCallback(check_freq=500, save_path=CHECKPOINT_DIR)
    model = DQNModel(
        'CnnPolicy', 
        env, 
        tensorboard_log=LOG_DIR, 
        verbose=1,   
        buffer_size=450000,   
        learning_starts= 250
    )
    model.learn(total_timesteps= 100000, callback=callback)
