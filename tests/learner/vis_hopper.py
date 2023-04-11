import gym
from gym import envs
import numpy as np
import time
from active_critic.utils.gym_utils import make_policy_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

if __name__ == '__main__':
    pd = make_policy_dict()
    asd = pd['push']
    expert, env_id = asd
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id]()
    env._freeze_rand_vec = False

    for i in range(10):
        done = False
        obs = env.reset()

        while not done:
            action = expert.get_action(obs)
            obs, rew, done, info = env.step(action)
            done = info['success'] == 1
            h = time.perf_counter()
            while time.perf_counter() - h < 0.5:
                env.render()