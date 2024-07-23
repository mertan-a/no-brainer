from gym import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import sys
sys.path.append('..')

import numpy as np
import os
from copy import deepcopy

class BASIC_ENVIRONMENT(EvoGymBase):
    """ in this environment, we either have a environment stimulus or not
    if we have an environment stimulus, the robot should move to the left
    if we do not have an environment stimulus, the robot should move to the right
    the sensing is done by the passive perception materials """

    def __init__(self, body, env_stimulus, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim, self.action_lookup = self.process_body(body)

        # make world
        self.world_length = int(world_json_path.split('_')[2].split('.')[0])
        init_pos = self.world_length // 2 - self.body.shape[1] // 2
        self.world = EvoWorld.from_json(os.path.join('environment/world_data', world_json_path))
        self.world.add_from_array('robot', self.body_to_sim, init_pos, 1, connections=connections) # robot is placed at the middle of the world horizontally

        # init sim
        EvoGymBase.__init__(self, self.world)

        # set viewer to track objects
        self.default_viewer.track_objects('robot')

        # open loop acting related variables
        self.actuation_frequency = actuation_frequency
        self.timestep = 0
        self.sinusoid = np.sin(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.sinusoid = 0.6 + (self.sinusoid+1) / 2.0
        self.cosine = np.cos(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.cosine = 0.6 + (self.cosine+1) / 2.0

        # save the environment stimulus
        self.env_stimulus = env_stimulus
        self.shrink_action = 0.6
        self.expand_action = 1.6
        self.no_shrink_action = 1.6
        self.no_expand_action = 0.6

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5,6]
            5,6 are mapped to 3
                5 shrinks when environmental stimulus is applied
                6 expands when environmental stimulus is applied
        in this environment, we only have passive perception materials
        these materials sense the environmental stimulus, IRREGARDLESS of where they are placed in the robot's body
        or to put it another way, the environmental stimulus is the same everywhere
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3

        # make a list for lookup when determining actions
        action_lookup = body.flatten()
        action_lookup = action_lookup[action_lookup != 0]

        return body, body_to_sim, action_lookup

    def step(self, _):
        if _ is not None:
            self.env_stimulus = _

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # assign action for each voxel
        action = []
        for i in range(self.action_lookup.shape[0]):
            if self.action_lookup[i] == 3:
                action.append(self.sinusoid[self.timestep])
            elif self.action_lookup[i] == 4:
                action.append(self.cosine[self.timestep])
            elif self.action_lookup[i] == 5:
                if self.env_stimulus:
                    action.append(self.shrink_action)
                else:
                    action.append(self.no_shrink_action)
            elif self.action_lookup[i] == 6:
                if self.env_stimulus:
                    action.append(self.expand_action)
                else:
                    action.append(self.no_expand_action)
        action = np.array(action)

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            return None, None, True, {} # TODO: does this make sense?

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
            
        # check goal met 
        if com_2[0] < 5 or com_2[0] > self.world_length - 5: # should never happen
            done = True

        # observation
        obs = None
        # keep track of the timestep
        self.timestep += 1
        self.timestep %= self.actuation_frequency

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        return None

class BASIC_ENVIRONMENT_TEST(EvoGymBase):
    """ test version designed to work with modified evogym simulator
    also we want to be able to change the environment stimulus on the fly at each step """

    def __init__(self, body, env_stimulus, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim, self.action_lookup = self.process_body(body)

        # make world
        self.world_length = int(world_json_path.split('_')[2].split('.')[0])
        init_pos = self.world_length // 2 - self.body.shape[1] // 2
        self.world = EvoWorld.from_json(os.path.join('environment/world_data', world_json_path))
        self.world.add_from_array('robot', self.body_to_sim, init_pos, 1, connections=connections, colors=self.body) # robot is placed at the middle of the world horizontally

        # init sim
        EvoGymBase.__init__(self, self.world)

        # set viewer to track objects
        self.default_viewer.track_objects('robot')

        # open loop acting related variables
        self.actuation_frequency = actuation_frequency
        self.timestep = 0
        self.sinusoid = np.sin(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.sinusoid = 0.6 + (self.sinusoid+1) / 2.0
        self.cosine = np.cos(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.cosine = 0.6 + (self.cosine+1) / 2.0

        # save the environment stimulus
        self.env_stimulus = env_stimulus
        self.shrink_action = 0.6
        self.expand_action = 1.6
        self.no_shrink_action = 1.6
        self.no_expand_action = 0.6

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5,6]
            5,6 are mapped to 3
                5 shrinks when environmental stimulus is applied
                6 expands when environmental stimulus is applied
        in this environment, we only have passive perception materials
        these materials sense the environmental stimulus, IRREGARDLESS of where they are placed in the robot's body
        or to put it another way, the environmental stimulus is the same everywhere
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3

        # make a list for lookup when determining actions
        action_lookup = body.flatten()
        action_lookup = action_lookup[action_lookup != 0]

        return body, body_to_sim, action_lookup

    def step(self, _):
        if _ is not None:
            self.env_stimulus = _

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # assign action for each voxel
        action = []
        for i in range(self.action_lookup.shape[0]):
            if self.action_lookup[i] == 3:
                action.append(self.sinusoid[self.timestep])
            elif self.action_lookup[i] == 4:
                action.append(self.cosine[self.timestep])
            elif self.action_lookup[i] == 5:
                if self.env_stimulus:
                    action.append(self.shrink_action)
                else:
                    action.append(self.no_shrink_action)
            elif self.action_lookup[i] == 6:
                if self.env_stimulus:
                    action.append(self.expand_action)
                else:
                    action.append(self.no_expand_action)
        action = np.array(action)

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            return None, None, True, {}

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
            
        # check goal met
        if com_2[0] < 5 or com_2[0] > self.world_length - 5:
            done = True

        # observation
        obs = None
        # keep track of the timestep
        self.timestep += 1
        self.timestep %= self.actuation_frequency

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        return None
