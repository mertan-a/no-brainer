from gym import spaces
from evogym import EvoWorld
from evogym.envs import EvoGymBase

import sys
sys.path.append('..')

import numpy as np
import os
from copy import deepcopy

class DIRECTIONAL_ENVIRONMENT(EvoGymBase):
    """ in this environment, we either have the environment stimulus diffusing from right hand-side or left hand-side
    we want robots to move towards the environmental stimulus
    the sensing is done by the passive perception materials """

    def __init__(self, body, env_stimulus, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim = self.process_body(body)

        # make world
        self.world = EvoWorld.from_json(os.path.join('environment/world_data', world_json_path))
        self.world.add_from_array('robot', self.body_to_sim, 28, 1, connections=connections) # robot is placed at the middle of the world horizontally

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
        if self.env_stimulus == "left":
            self.expand_action = np.linspace(1.6, 0.6, self.body.shape[1])
            self.shrink_action = np.linspace(0.6, 1.6, self.body.shape[1])
        elif self.env_stimulus == "right":
            self.expand_action = np.linspace(0.6, 1.6, self.body.shape[1])
            self.shrink_action = np.linspace(1.6, 0.6, self.body.shape[1])
        else:
            ...

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5, 6]
            5, 6 are mapped to 3
                5 shrinks propostional to the environmental stimulus
                6 expands proportional to the environmental stimulus
        in this environment, we only have passive perception materials
        these materials sense the environmental stimulus and response to them based on where the environmental stimulus is coming from AND how they are placed in a body
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3
        return body, body_to_sim

    def step(self, _):

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
                    position = 0
                    if self.env_stimulus == "left":
                        for i in range(col):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.shrink_action[position]
                    elif self.env_stimulus == "right":
                        for i in range(col+1, self.body.shape[1]):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.shrink_action[self.body.shape[1]-position-1]
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 6:
                    position = 0
                    if self.env_stimulus == "left":
                        for i in range(col):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.expand_action[position]
                    elif self.env_stimulus == "right":
                        for i in range(col+1, self.body.shape[1]):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.expand_action[self.body.shape[1]-position-1]
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
            reward = 3.0
            return None, reward, True, {}

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        if self.env_stimulus == "left":
            reward = (com_1[0] - com_2[0])
        elif self.env_stimulus == "right":
            reward = (com_2[0] - com_1[0])
        else:
            reward = np.abs(com_2[0] - com_1[0]) * -1
            
        # check goal met
        if self.env_stimulus == "left" and com_2[0] < 2:
            done = True
            reward += 1.0
        elif self.env_stimulus == "right" and com_2[0] > 58:
            done = True
            reward += 1.0

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


class DIRECTIONAL_ENVIRONMENT_TEST(EvoGymBase):
    """ in this environment, we either have the environment stimulus diffusing from right hand-side or left hand-side
    we want robots to move towards the environmental stimulus
    the sensing is done by the passive perception materials """

    def __init__(self, body, env_stimulus, actuation_frequency, world_json_path, connections=None):
        # process the body
        self.body, self.body_to_sim = self.process_body(body)

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
        if self.env_stimulus == "left":
            self.expand_action = np.linspace(1.6, 0.6, self.body.shape[1])
            self.shrink_action = np.linspace(0.6, 1.6, self.body.shape[1])
        elif self.env_stimulus == "right":
            self.expand_action = np.linspace(0.6, 1.6, self.body.shape[1])
            self.shrink_action = np.linspace(1.6, 0.6, self.body.shape[1])
        else:
            ...

    def process_body(self, body):
        '''
        0 means empty
        original materials \in [1,2,3]
            3 are active materials controlled by the sinusoidal signal
        antiphase active material \in [4]
            4 is mapped to 3
            4 is an active material controlled by the cosine signal
        mechanical passive perception materials \in [5, 6]
            5, 6 are mapped to 3
                5 shrinks propostional to the environmental stimulus
                6 expands proportional to the environmental stimulus
        in this environment, we only have passive perception materials
        these materials sense the environmental stimulus and response to them based on where the environmental stimulus is coming from AND how they are placed in a body
        '''
        # prepare a body for the simulation
        body_to_sim = deepcopy(body)
        body_to_sim[body_to_sim > 3] = 3
        return body, body_to_sim

    def step(self, _):
        if _ is not None:
            self.env_stimulus = _
            if self.env_stimulus == "left":
                self.expand_action = np.linspace(1.6, 0.6, self.body.shape[1])
                self.shrink_action = np.linspace(0.6, 1.6, self.body.shape[1])
            elif self.env_stimulus == "right":
                self.expand_action = np.linspace(0.6, 1.6, self.body.shape[1])
                self.shrink_action = np.linspace(1.6, 0.6, self.body.shape[1])
            else:
                ...

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
                    position = 0
                    if self.env_stimulus == "left":
                        for i in range(col):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.shrink_action[position]
                    elif self.env_stimulus == "right":
                        for i in range(col+1, self.body.shape[1]):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.shrink_action[self.body.shape[1]-position-1]
                    else:
                        action[row,col] = 1.6
                elif self.body[row,col] == 6:
                    position = 0
                    if self.env_stimulus == "left":
                        for i in range(col):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.expand_action[position]
                    elif self.env_stimulus == "right":
                        for i in range(col+1, self.body.shape[1]):
                            if self.body[row,i] != 0:
                                position += 1
                        action[row,col] = self.expand_action[self.body.shape[1]-position-1]
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
            reward = 0.0
            return None, reward, True, {}

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        if self.env_stimulus == "left":
            reward = (com_1[0] - com_2[0])
        elif self.env_stimulus == "right":
            reward = (com_2[0] - com_1[0])
        else:
            reward = np.abs(com_2[0] - com_1[0]) * -1
            
        # check goal met
        if com_2[0] < 2 or com_2[0] > self.world_length - 2:
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
