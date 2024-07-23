from gym import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import sys
sys.path.append('..')

import numpy as np
import os
from copy import deepcopy

class LOGIC_ENVIRONMENT(EvoGymBase):
    """ this is a class to generate multiple environments
    each environment requires robot to realize a different logic gate
    robot senses what would be the input to the logic gate and then it should move to the correct position to realize the logic gate
    i.e. OR gate, inputs 0,0 -> output 0: robot should move to the left; inputs 0,1 -> output 1: robot should move to the right
    the sensing is done by the passive perception materials """

    def __init__(self, body, inputs, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim = self.process_body(body)

        # save the inputs
        self.inputs = inputs

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
        self.cosine = np.sin(np.linspace(1*np.pi, 3*np.pi, self.actuation_frequency))
        self.cosine = 0.6 + (self.cosine+1) / 2.0

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5, 6, 7, 8]
            5, 6 are mapped to 3
                5 shrinks if inputs[0] is 1
                6 shrinks if inputs[1] is 1
            7, 8 are mapped to 3
                7 expands if inputs[0] is 1
                8 expands if inputs[1] is 1
        in this environment, we only have passive perception materials
        these materials sense the inputs to the logic gate and response to them IRREGARDLESS of where they are placed in a body
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3
        return body, body_to_sim

    def step(self, _):
        if _ is not None:
            self.inputs = _

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # assign action for each voxel
        action = np.ones_like(self.body) * -1.0 #for debuggin purposes
        for row in range(self.body.shape[0]):
            for col in range(self.body.shape[1]):
                if self.body[row,col] == 3:
                    action[row,col] = self.sinusoid[self.timestep]
                elif self.body[row,col] == 4:
                    action[row,col] = self.cosine[self.timestep]
                elif self.body[row,col] == 5:
                    if self.inputs[0] == 1:
                        action[row,col] = 0.6
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 6:
                    if self.inputs[1] == 1:
                        action[row,col] = 0.6
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 7:
                    if self.inputs[0] == 1:
                        action[row,col] = 1.6
                    else:
                        action[row,col] = 0.6
                elif self.body[row,col] == 8:
                    if self.inputs[1] == 1:
                        action[row,col] = 1.6
                    else:
                        action[row,col] = 0.6
        action = action[self.body > 2]
        action = action.flatten()

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

class LOGIC_ENVIRONMENT_TEST(EvoGymBase):
    """ this is a class to generate multiple environments
    each environment requires robot to realize a different logic gate
    robot senses what would be the input to the logic gate and then it should move to the correct position to realize the logic gate
    i.e. OR gate, inputs 0,0 -> output 0: robot should move to the left; inputs 0,1 -> output 1: robot should move to the right
    the sensing is done by the passive perception materials """

    def __init__(self, body, inputs, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim = self.process_body(body)

        # save the inputs
        self.inputs = inputs

        # CORRECTION FOR COLORS FOR TEST PURPOSES ONLY swap 6 and 7 for the test environment for color
        colors = np.zeros_like(self.body)
        colors = np.copy(self.body)
        colors[self.body == 6] = 7
        colors[self.body == 7] = 6
        # make world
        self.world_length = int(world_json_path.split('_')[2].split('.')[0])
        init_pos = self.world_length // 2 - self.body.shape[1] // 2
        self.world = EvoWorld.from_json(os.path.join('environment/world_data', world_json_path))
        self.world.add_from_array('robot', self.body_to_sim, init_pos, 1, connections=connections, colors=colors) # robot is placed at the middle of the world horizontally

        # init sim
        EvoGymBase.__init__(self, self.world)

        # set viewer to track objects
        #self.default_viewer.track_objects('robot')

        # open loop acting related variables
        self.actuation_frequency = actuation_frequency
        self.timestep = 0
        self.sinusoid = np.sin(np.linspace(0, 2*np.pi, self.actuation_frequency))
        self.sinusoid = 0.6 + (self.sinusoid+1) / 2.0
        self.cosine = np.sin(np.linspace(1*np.pi, 3*np.pi, self.actuation_frequency))
        self.cosine = 0.6 + (self.cosine+1) / 2.0

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5, 6, 7, 8]
            5, 6 are mapped to 3
                5 shrinks if inputs[0] is 1
                6 shrinks if inputs[1] is 1
            7, 8 are mapped to 3
                7 expands if inputs[0] is 1
                8 expands if inputs[1] is 1
        in this environment, we only have passive perception materials
        these materials sense the inputs to the logic gate and response to them IRREGARDLESS of where they are placed in a body
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3
        return body, body_to_sim

    def step(self, _):
        if _ is not None:
            self.inputs = _

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # assign action for each voxel
        action = np.ones_like(self.body) * -1.0 #for debuggin purposes
        for row in range(self.body.shape[0]):
            for col in range(self.body.shape[1]):
                if self.body[row,col] == 3:
                    action[row,col] = self.sinusoid[self.timestep]
                elif self.body[row,col] == 4:
                    action[row,col] = self.cosine[self.timestep]
                elif self.body[row,col] == 5:
                    if self.inputs[0] == 1:
                        action[row,col] = 0.6
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 6:
                    if self.inputs[1] == 1:
                        action[row,col] = 0.6
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 7:
                    if self.inputs[0] == 1:
                        action[row,col] = 1.6
                    else:
                        action[row,col] = 0.6
                elif self.body[row,col] == 8:
                    if self.inputs[1] == 1:
                        action[row,col] = 1.6
                    else:
                        action[row,col] = 0.6
        action = action[self.body > 2]
        action = action.flatten()

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

