import string
from copy import deepcopy
from datetime import datetime
import random
import numpy as np


class INDIVIDUAL():
    ''' individual class that contains brain and body '''

    def __init__(self, body):
        ''' initialize the individual class with the given body and brain

        Parameters
        ----------
        body: BODY class instance or list of BODY class instances
            defines the shape and muscle properties of the robot

        '''
        # ids
        self.parent_id = ''
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        # attributes
        self.fitness_variable = None
        self.fitness_fixed = None
        self.behavior = None
        self.cum_rewards_variable_stim = None
        self.cum_rewards_fixed_stim = None
        self.age = 0
        # initialize main components
        self.body = body

    def mutate(self):
        # handle ids
        self.parent_id = self.self_id
        self.self_id = ''.join(random.sample(string.ascii_uppercase, k=5))
        self.self_id += '_' + datetime.now().strftime('%H%M%S%f')
        self.body.mutate()
        return

    def produce_offspring(self):
        '''
        produce an offspring from the current individual
        '''
        while True:
            offspring = deepcopy(self)
            offspring.mutate()
            if offspring.is_valid():
                break
        offspring.fitness_variable = None
        offspring.fitness_fixed = None
        offspring.behavior = None
        offspring.cum_rewards_variable_stim = None
        offspring.cum_rewards_fixed_stim = None
        return offspring

    def is_valid(self):
        '''
        check if the individual is valid
        '''
        if self.body.is_valid():
            return True
        else:
            return False

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        new = self.__class__(body=self.body)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

