from active_critic.utils.gym_utils import make_policy_dict
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

def vis(name):
    pd = make_policy_dict()
    expert, envid = pd[name]
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[envid]()
    obsv = env.reset()
    done = False
    while not done:
        action = expert.get_action(obsv)
        obsv, rew, done, info = env.step(action)
        for i in range(50):
            env.render()
        print(rew)


if __name__ == '__main__':
    vis('plateslide')