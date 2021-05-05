#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import gym 
from stable_baselines import ACER, A2C, DQN, HER, TD3
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

environment_name = 'MountainCar-v0'

env = gym.make(environment_name)
episodes = 2
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


env = gym.make(environment_name)
env = DummyVecEnv([lambda: env]) # loops through the env and converts it into dummy vectors
model = ACER('MlpPolicy', env, verbose = 1)

model.learn(total_timesteps=100000)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

model.save("ACER_model")

del model

model = ACER.load("ACER_model", env=env)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()



# In[ ]:





# In[ ]:




