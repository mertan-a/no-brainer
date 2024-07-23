import multiprocessing
import gym
import numpy as np

from evogym.envs import *
from evogym import get_full_connectivity
import environment.envs as envs

def get_sim_pairs(population, **kwargs):
    sim_pairs = []
    for idx, ind in enumerate(population):
        sim_pairs.append( {'ind':ind, 'kwargs':kwargs} )
    return sim_pairs

def simulate_ind(sim_pair):
    # unpack the simulation pair
    ind = sim_pair['ind']
    kwargs = sim_pair['kwargs']
    # check if the individual has fitness already assigned (e.g. from previous subprocess run. sometimes process hangs and does not return, all the population is re-submitted to the queue)
    if ind.cum_rewards_variable_stim is not None:
        return ind, cum_rewards_variable_stim, cum_rewards_fixed_stim
    # otherwise, simulate the individual
    if kwargs['task'] == 'BasicEnv-v0':
        body = ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        #### single simulation with changing stimulus
        stimulus = False
        env = gym.make('BasicEnv-v0', body=body, env_stimulus=stimulus, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        env.seed(17)
        # run the environment
        cum_rewards_variable_stim = {False:0, True:0}
        _ = env.reset()
        for ts in range(1680):
            if ts < 840:
                stimulus = False
            else:
                stimulus = True
            _, reward, done, _ = env.step(stimulus)
            if reward is None:
                return ind, None, None
            cum_rewards_variable_stim[stimulus] += reward
        env.close()

        #### multiple simulations with fixed stimulus
        cum_rewards_fixed_stim = {False:0, True:0}
        for stimulus in [False, True]:
            env = gym.make('BasicEnv-v0', body=body, env_stimulus=stimulus, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
            env.seed(17)
            # run the environment
            _ = env.reset()
            for ts in range(840):
                _, reward, done, _ = env.step(None)
                if reward is None:
                    return ind, None, None
                cum_rewards_fixed_stim[stimulus] += reward
            env.close()

    elif kwargs['task'] == 'LogicEnv-v0':
        body = ind.body.to_phenotype()
        connections = get_full_connectivity(body)
        #### single simulation with changing stimulus
        input = (0,0)
        env = gym.make('LogicEnv-v0', body=body, inputs=input, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
        env.seed(17)
        # run the environment
        cum_rewards_variable_stim = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        _ = env.reset()
        for ts in range(3360):
            if ts < 840:
                input = (0,0)
            elif ts < 1680:
                input = (0,1)
            elif ts < 2520:
                input = (1,0)
            else:
                input = (1,1)
            _, reward, done, _ = env.step(input)
            if reward is None:
                return ind, None, None
            cum_rewards_variable_stim[input] += reward

        #### multiple simulations with fixed stimulus
        cum_rewards_fixed_stim = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        for input in [(0,0), (0,1), (1,0), (1,1)]: # all possible inputs
            env = gym.make('LogicEnv-v0', body=body, inputs=input, actuation_frequency=84, world_json_path='flat_env_500.json', connections=connections)
            env.seed(17)
            # run the environment
            _ = env.reset()
            for ts in range(840):
                _, reward, done, _ = env.step(input)
                if reward is None:
                    return ind, None, None
                cum_rewards_fixed_stim[input] += reward
            env.close()

    else:
        raise NotImplementedError
    return ind, cum_rewards_variable_stim, cum_rewards_fixed_stim

def simulate_population(population, **kwargs):
    #get the simulator 
    sim_pairs = get_sim_pairs(population, **kwargs)
    # run the simulation
    finished = False
    while not finished:
        with multiprocessing.Pool(processes=len(sim_pairs)) as pool:
            results_f = pool.map_async(simulate_ind, sim_pairs)
            try:
                results = results_f.get(timeout=580)
                finished = True
            except multiprocessing.TimeoutError:
                print('TimeoutError')
                pass
    # assign fitness
    for r in results:
        ind, cum_rewards_variable_stim, cum_rewards_fixed_stim = r
        for i in population:
            if i.self_id == ind.self_id:
                i.cum_rewards_variable_stim = cum_rewards_variable_stim
                i.cum_rewards_fixed_stim = cum_rewards_fixed_stim




