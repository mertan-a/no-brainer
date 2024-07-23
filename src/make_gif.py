import gym
import numpy as np
np.float = float
import matplotlib.pyplot as plt
import imageio

from evogym.envs import *
import environment.envs as envs
from evogym import get_full_connectivity
from utils import RenderWrapper

class MAKEGIF():

    def __init__(self, args, ind, output_path):
        self.kwargs = vars(args)
        self.ind = ind
        self.output_path = output_path

    def run(self):
        if self.kwargs['task'] == 'BasicEnv-v0':
            imgs = self.run_basic_env()
            return imgs
        elif self.kwargs['task'] == 'LogicEnv-v0':
            imgs = self.run_logic_env()
            return imgs
        else:
            raise NotImplementedError

    def run_basic_env(self):
        body = self.ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        imgs = []
        if 'local' in self.kwargs.keys() and self.kwargs['local']:
            env = gym.make('BasicEnvTest-v0', body=body, env_stimulus=False, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        else:
            env = gym.make('BasicEnv-v0', body=body, env_stimulus=False, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        env = RenderWrapper(env, render_mode='img')
        env.seed(17)

        # run the environment
        _ = env.reset()
        for ts in range(1680):
            if ts < 840:
                _, reward, done, _ = env.step(None)
            else:
                _, reward, done, _ = env.step(True)
        imageio.mimsave(f"{self.output_path}variablef_{self.ind.fitness_variable}_fixedf_{self.ind.fitness_fixed}_behavior_{self.ind.behavior}.gif", env.imgs[0::6], duration=1.0/100)
        imgs.append(env.imgs)
        return imgs

    def run_logic_env(self):
        body = self.ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        imgs = []
        if 'local' in self.kwargs.keys() and self.kwargs['local']:
            env = gym.make('LogicEnvTest-v0', body=body, inputs=[0,0], actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        else:
            env = gym.make('LogicEnv-v0', body=body, inputs=[0,0], actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        env = RenderWrapper(env, render_mode='img')
        env.seed(17)

        # run the environment
        _ = env.reset()
        for ts in range(3360):
            if ts < 840:
                _, reward, done, _ = env.step(None)
            elif ts < 1680:
                _, reward, done, _ = env.step([0,1])
            elif ts < 2520:
                _, reward, done, _ = env.step([1,0])
            else:
                _, reward, done, _ = env.step([1,1])
        imageio.mimsave(f"{self.output_path}variablef_{self.ind.fitness_variable}_fixedf_{self.ind.fitness_fixed}_behavior_{self.ind.behavior}.gif", env.imgs[0::6], duration=1.0/100)
        imgs.append(env.imgs)
        return imgs
