import gym
from gym import envs
import numpy as np

if __name__ == '__main__':
    env = gym.make('Hopper-v3')
    result = env.reset()
    for i in range(300):
        obs, rew, done, info = env.step(np.array([1,1,0]))
        for i in range(100):
            env.render()
        print(rew)
        print(done)