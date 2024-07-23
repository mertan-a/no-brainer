import gym
from evogym import sample_robot, get_full_connectivity
import numpy as np
np.float = float
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

# import envs from the envs folder and register them
import envs

if __name__ == '__main__':

    # create a random robot
    body = np.array( [ [4,4,6,5],
                       [3,3,6,5] ] )
    connections = get_full_connectivity(body)

    env = gym.make('BasicEnv-v0', body=body, env_stimulus=True, actuation_frequency=28, world_json_path='flat_env_60.json', connections=connections)
    env.reset()

    # step the environment for 500 iterations
    cum_reward = 0
    for i in range(1000):

        ob, reward, done, info = env.step(action=None)
        env.render(verbose=True)
        print(f'Step: {i}, Reward: {reward}, Done: {done}')
        cum_reward += reward

        if done:
            env.reset()
    print(f'Cumulative Reward: {cum_reward}')

    env.close()
