import operator
import numpy as np

from individual import INDIVIDUAL
from body import CPPN_BODY


class POPULATION(object):
    """A population of individuals"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.individuals = []
        self.non_dominated_size = 0

        while len(self) < self.args.nr_parents:
            self.add_individual()

    def add_individual(self):
        valid = False
        while not valid:
            # body
            body = CPPN_BODY(self.args)
            ind = INDIVIDUAL(body=body)
            if ind.is_valid():
                self.individuals.append(ind)
                valid = True

    def produce_offsprings(self):
        """Produce offspring from the current population."""
        offspring = []
        for counter, ind in enumerate(self.individuals):
            offspring.append(ind.produce_offspring())
        self.individuals.extend(offspring)

    def calc_dominance(self):
        """Determine which other individuals in the population dominate each individual."""

        # if tied on all objectives, give preference to newer individual
        self.sort(key="age", reverse=False)

        # clear old calculations of dominance
        self.non_dominated_size = 0
        for ind in self:
            ind.dominated_by = []
            ind.pareto_level = 0

        for ind in self:
            for other_ind in self:
                if other_ind.self_id != ind.self_id:
                    if self.dominated_in_multiple_objectives(ind, other_ind) and (ind.self_id not in other_ind.dominated_by):
                        ind.dominated_by += [other_ind.self_id]

            ind.pareto_level = len(ind.dominated_by)  # update the pareto level

            # update the count of non_dominated individuals
            if ind.pareto_level == 0:
                self.non_dominated_size += 1

    def dominated_in_multiple_objectives(self, ind1, ind2):
        """Calculate if ind1 is dominated by ind2 according to all objectives in objective_dict.

        If ind2 is better or equal to ind1 in all objectives, and strictly better than ind1 in at least one objective.

        """
        wins = []  # 1 dominates 2
        wins += [ind1.fitness > ind2.fitness]
        wins += [ind1.age < ind2.age]
        return not np.any(wins)

    def sort_by_objectives(self):
        """Sorts the population multiple times by each objective, from least important to most important."""
        self.sort(key="age", reverse=False)
        self.sort(key="fitness", reverse=True)

        self.sort(key="pareto_level", reverse=False)  # min

    def update_ages(self):
        """Increment the age of each individual."""
        for ind in self:
            ind.age += 1

    def sort(self, key, reverse=False):
        """Sort individuals by their attributes.

        Parameters
        ----------
        key : str
            An individual-level attribute.

        reverse : bool
            True sorts from largest to smallest (useful for maximizing an objective).
            False sorts from smallest to largest (useful for minimizing an objective).

        """
        return self.individuals.sort(reverse=reverse, key=operator.attrgetter(key))

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        return iter(self.individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        try:
            return n in self.individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        return len(self.individuals)

    def __getitem__(self, n):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.individuals[n]

    def pop(self, index=None):
        """Remove and return item at index (default last)."""
        return self.individuals.pop(index)

    def append(self, individuals):
        """Append a list of new individuals to the end of the population.

        Parameters
        ----------
        individuals : list of/or INDIVIDUAL
            A list of individuals to append or a single INDIVIDUAL to append

        """
        if type(individuals) == list:
            for n in range(len(individuals)):
                if type(individuals[n]) != INDIVIDUAL:
                    raise TypeError("Non-INDIVIDUAL added to the population")
            self.individuals += individuals
        elif type(individuals) == INDIVIDUAL:
            self.individuals += [individuals]
        else:
            raise TypeError("Non-INDIVIDUAL added to the population")


class ARCHIVE():
    """A population of individuals to be used with MAP-Elites"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.map = {}
        # define the bins
        # first dimensions are based on body shape
        self.n_bins_existing_voxels = 4
        self.n_bins_active_voxels = 4
        self.n_bins_perception_voxels = 4
        # other dimensions are based on behavior
        self.stim_1_p = ['left', 'right'] # which way the robot moved when the stimulus was present
        self.stim_1_a = ['left', 'right'] # which way the robot moved when the stimulus was absent
        self.stim_2_p = ['left', 'right', None] # there may or may not be a second stimulus
        self.stim_2_a = ['left', 'right', None]
        for i in range(1, self.n_bins_existing_voxels+1):
            for j in range(1, self.n_bins_active_voxels+1):
                for k in range(1, self.n_bins_perception_voxels+1):
                    for l in self.stim_1_p:
                        for m in self.stim_1_a:
                            for n in self.stim_2_p:
                                for o in self.stim_2_a:
                                    self.map[(i,j,k,l,m,n,o)] = None

    def get_random_individual(self):
        valid = False
        while not valid:
            # body
            body = CPPN_BODY(self.args)
            ind = INDIVIDUAL(body=body)
            if ind.is_valid():
                valid = True
        return ind

    def produce_offsprings(self):
        """Produce offspring from the current map."""
        # check if there are any individuals in the map
        if len(self) == 0:
            init_population = []
            while len(init_population) < self.args.nr_parents:
                init_population.append(self.get_random_individual())
            return init_population
        # choose nr_parents many random keys from the map. make sure that they are not None
        valid_keys = [ k for k in self.map.keys() if self.map[k] is not None ]
        nr_valid_keys = len(valid_keys) if len(valid_keys) < self.args.nr_parents else self.args.nr_parents
        random_keys_idx = np.random.choice(len(valid_keys), size=nr_valid_keys, replace=False)
        # produce offsprings
        offsprings = []
        for key_idx in random_keys_idx:
            key = valid_keys[key_idx]
            offsprings.append(self.map[key].produce_offspring())
        return offsprings

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return iter(individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        try:
            return n in individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return len(individuals)

    def __getitem__(self, x, y):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.map[(x,y)]

    def get_best_individual(self):
        """Return the best individual in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness_variable'))

    def get_best_fitness(self):
        """Return the best fitness in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness_variable')).fitness_variable

    def update_map(self, population):
        """Update the map with the given population."""
        for ind in population:
            # continue if the individual is not valid
            if ind.fitness_variable is None:
                continue
            # determine the bins
            bin = self.determine_bins(ind)
            # update the map
            if self.map[bin] is None:
                self.map[bin] = ind
            else:
                current_fitness_variable = self.map[bin].fitness_variable
                current_fitness_fixed = self.map[bin].fitness_fixed
                if ind.fitness_variable > current_fitness_variable and ind.fitness_fixed > current_fitness_fixed:
                    self.map[bin] = ind

    def determine_bins(self, ind):
        """Calculate the bin indices for the given individual."""
        # shape based bins
        existing_voxels = ind.body.count_existing_voxels();
        active_voxels = ind.body.count_active_voxels();
        perception_voxels = ind.body.count_perception_voxels();

        nr_voxel_per_bin = (self.args.bounding_box[0] * self.args.bounding_box[1]) // 4
        bin_existing_voxels = (existing_voxels // nr_voxel_per_bin) + 1; bin_existing_voxels -= 1 if bin_existing_voxels == 5 else 0
        bin_active_voxels = (active_voxels // nr_voxel_per_bin) + 1; bin_active_voxels -= 1 if bin_active_voxels == 5 else 0
        bin_perception_voxels = (perception_voxels // nr_voxel_per_bin) + 1; bin_perception_voxels -= 1 if bin_perception_voxels == 5 else 0

        # behavior based bins
        assert len(ind.behavior) == 2 or len(ind.behavior) == 4
        if len(ind.behavior) == 2:
            behavior_1 = ind.behavior[0]
            behavior_2 = ind.behavior[1]
            behavior_3 = None
            behavior_4 = None
        else:
            behavior_1 = ind.behavior[0]
            behavior_2 = ind.behavior[1]
            behavior_3 = ind.behavior[2]
            behavior_4 = ind.behavior[3]

        return (bin_existing_voxels, bin_active_voxels, bin_perception_voxels, behavior_1, behavior_2, behavior_3, behavior_4)

    def print_map(self):
        """Print some useful information about the map."""
        # print the best fitness in the map
        print("Best fitness in the map: ", self.get_best_individual().fitness_variable)
        # print the occupancy of the map
        print("Occupancy of the map: ", len(self), "/", len(self.map))

    def get_fitnesses(self, fitness_type='variable'):
        """return a numpy array of fitnesses of the individuals in the map,
        with a mask to indicate which bins are not empty"""
        fitnesses = np.zeros((self.n_bins_existing_voxels, self.n_bins_active_voxels, self.n_bins_perception_voxels, len(self.stim_1_p), len(self.stim_1_a), len(self.stim_2_p), len(self.stim_2_a)))
        for i in range(1, self.n_bins_existing_voxels+1):
            for j in range(1, self.n_bins_active_voxels+1):
                for k in range(1, self.n_bins_perception_voxels+1):
                    for il, l in enumerate(self.stim_1_p):
                        for im, m in enumerate(self.stim_1_a):
                            for in_, n in enumerate(self.stim_2_p):
                                for io, o in enumerate(self.stim_2_a):
                                    if self.map[(i,j,k,l,m,n,o)] is not None:
                                        if fitness_type == 'variable':
                                            fitnesses[i-1,j-1,k-1,il,im,in_,io] = self.map[(i,j,k,l,m,n,o)].fitness_variable
                                        else:
                                            fitnesses[i-1,j-1,k-1,il,im,in_,io] = self.map[(i,j,k,l,m,n,o)].fitness_fixed
                                    else:
                                        fitnesses[i-1,j-1,k-1,il,im,in_,io] = -9999
        # mask for non-empty bins
        mask = fitnesses != -9999
        return fitnesses, mask



        

