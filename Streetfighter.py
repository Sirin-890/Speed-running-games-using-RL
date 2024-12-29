import numpy as np
import os 
import retro
from gym import Env
from gym.spaces import Box, MultiBinary
import cv2
from matplotlib import pyplot as plt
import optuna
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
logger.error(retro.data.list_games())
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
#env.close()
env.observation_space
env.action_space.sample()
obs = env.reset()
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        time.sleep(0.01)
        print(reward)
env.close()
print(info)             
class StreetFighter(Env): 
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs) 
        self.previous_frame = obs 
        
        self.score = 0 
        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84,84,1))
        return channels 
    
    def step(self, action): 
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs) 
        
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs 
        
        reward = info['score'] - self.score 
        self.score = info['score'] 
        
        return frame_delta, reward, done, info
    def render(self, *args, **kwargs):
        self.game.render()
        
    def close(self):
        self.game.close()
env = StreetFighter()
env.observation_space.shape
env.action_space.shape
obs = env.reset()
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        time.sleep(0.01)
        if reward > 0: 
            print(reward)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
plt.imshow(cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
log_dir='logs'
opt_dir='opt'
def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }
SAVE_PATH = os.path.join(opt_dir, 'trial_{}_best_model'.format(1))
def optimize_agent(trial):
    try:
        model_params = optimize_ppo(trial) 

        env = StreetFighter()
        env = Monitor(env, log_dir)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        model = PPO('CnnPolicy', env, tensorboard_log=log_dir, verbose=0, **model_params)
        model.learn(total_timesteps=30000)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        SAVE_PATH = os.path.join(opt_dir, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -1000
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=10, n_jobs=1)
logger.info(study.best_params)
logger.debug(study.best_trial)
model = PPO.load(os.path.join(opt_dir, 'trial_5_best_model.zip'))
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
check_dir='train'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=check_dir)
env = StreetFighter()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')
model_params = study.best_params
model_params['n_steps'] = 7488  
model_params
model = PPO('CnnPolicy', env, tensorboard_log=log_dir, verbose=1, **model_params)
model.load(os.path.join(opt_dir, 'trial_5_best_model.zip'))
model.learn(total_timesteps=100000, callback=callback)
model = PPO.load('./opt/trial_5_best_model.zip')
mean_reward, _ = evaluate_policy(model, env, render=True, n_eval_episodes=1)
logger.debug(mean_reward)
obs = env.reset()
logger.info(obs.shape)
env.step(model.predict(obs)[0])
obs = env.reset()
done = False
for game in range(1): 
    while not done: 
        if done: 
            obs = env.reset()
        env.render()
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        time.sleep(0.01)
        logger.debug(reward)