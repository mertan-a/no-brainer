import numpy as np
import os
from copy import deepcopy
import inspect
import random
import re
import networkx as nx
from networkx import DiGraph
from collections import OrderedDict

#### math functions
def identity(x):
    return x

def sigmoid(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0

def positive_sigmoid(x):
    return (1 + sigmoid(x)) * 0.5

def rescaled_positive_sigmoid(x, x_min=0, x_max=1):
    return (x_max - x_min) * positive_sigmoid(x) + x_min

def inverted_sigmoid(x):
    return sigmoid(x) ** -1

def neg_abs(x):
    return -np.abs(x)

def neg_square(x):
    return -np.square(x)

def sqrt_abs(x):
    return np.sqrt(np.abs(x))

def neg_sqrt_abs(x):
    return -sqrt_abs(x)

def mean_abs(x):
    return np.mean(np.abs(x))

def std_abs(x):
    return np.std(np.abs(x))

def count_positive(x):
    return np.sum(np.greater(x, 0))

def count_negative(x):
    return np.sum(np.less(x, 0))

def normalize(x):
    x -= np.min(x)
    x /= np.max(x)
    x = np.nan_to_num(x)
    x *= 2
    x -= 1
    return x

#### cppn related
class GenotypeToPhenotypeMap(object):
    """A mapping of the relationship from genotype (networks) to phenotype (VoxCad simulation)."""

    # TODO: generalize dependencies from boolean to any operation (e.g. to set an env param from multiple outputs)

    def __init__(self):
        self.mapping = dict()
        self.dependencies = dict()

    def items(self):
        """to_phenotype_mapping.items() -> list of (key, value) pairs in mapping"""
        return [(key, self.mapping[key]) for key in self.mapping]

    def __contains__(self, key):
        """Return True if key is a key str in the mapping, False otherwise. Use the expression 'key in mapping'."""
        try:
            return key in self.mapping
        except TypeError:
            return False

    def __len__(self):
        """Return the number of mappings. Use the expression 'len(mapping)'."""
        return len(self.mapping)

    def __getitem__(self, key):
        """Return mapping for node with name 'key'.  Use the expression 'mapping[key]'."""
        return self.mapping[key]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def add_map(self, name, func=sigmoid, output_type=float, dependency_order=None, params=None, param_tags=None,
                env_kws=None, logging_stats=np.mean):
        """Add an association between a genotype output and a VoxCad parameter.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        func : func
            Specifies relationship between attributes and xml tag.

        output_type : type
            The output type

        dependency_order : list
            Order of operations

        params : list
            Constants dictating parameters of the mapping

        param_tags : list
            Tags for any constants associated with the mapping

        env_kws : dict
            Specifies which function of the output state to use (on top of func) to set an Env attribute

        logging_stats : func or list
            One or more functions (statistics) of the output to be logged as additional column(s) in logging

        """
        if (dependency_order is not None) and not isinstance(dependency_order, list):
            dependency_order = [dependency_order]

        if params is not None:
            assert (param_tags is not None)
            if not isinstance(params, list):
                params = [params]

        if param_tags is not None:
            assert (params is not None)
            if not isinstance(param_tags, list):
                param_tags = [param_tags]
            param_tags = [xml_format(t) for t in param_tags]

        if (env_kws is not None) and not isinstance(env_kws, dict):
            env_kws = {env_kws: np.mean}

        if (logging_stats is not None) and not isinstance(logging_stats, list):
            logging_stats = [logging_stats]

        self.mapping[name] = {"func": func,
                              "dependency_order": dependency_order,
                              "state": None,
                              "old_state": None,
                              "output_type": output_type,
                              "params": params,
                              "param_tags": param_tags,
                              "env_kws": env_kws,
                              "logging_stats": logging_stats}

    def add_output_dependency(self, name, dependency_name, requirement, material_if_true=None, material_if_false=None):
        """Add a dependency between two genotype outputs.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        dependency_name : str
            Another network output node name.

        requirement : bool
            Dependency must be this

        material_if_true : int
            The material if dependency meets pre-requisite

        material_if_false : int
            The material otherwise

        """
        self.dependencies[name] = {"depends_on": dependency_name,
                                   "requirement": requirement,
                                   "material_if_true": material_if_true,
                                   "material_if_false": material_if_false,
                                   "state": None}

    def get_dependency(self, name, output_bool):
        """Checks recursively if all boolean requirements were met in dependent outputs."""
        if self.dependencies[name]["depends_on"] is not None:
            dependency = self.dependencies[name]["depends_on"]
            requirement = self.dependencies[name]["requirement"]
            return np.logical_and(self.get_dependency(dependency, True) == requirement,
                                  self.dependencies[name]["state"] == output_bool)
        else:
            return self.dependencies[name]["state"] == output_bool

def calc_outputs(network, orig_size_xyz, itself):
    """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""

    for name in network.graph.nodes():
        # flag all nodes as unevaluated
        network.graph.nodes[name]["evaluated"] = False

    network.set_input_node_states(orig_size_xyz)  # reset the inputs

    for name in network.output_node_names:
        network.graph.nodes[name]["state"] = np.zeros(
            orig_size_xyz)  # clear old outputs
        network.graph.nodes[name]["state"] = calc_node_state(network, orig_size_xyz, name)  # calculate new outputs

def calc_node_state(network, orig_size_xyz, node_name):
    """Propagate input values through the network"""
    if network.graph.nodes[node_name]["evaluated"]:
        return network.graph.nodes[node_name]["state"]

    network.graph.nodes[node_name]["evaluated"] = True
    input_edges = network.graph.in_edges(nbunch=[node_name])
    new_state = np.zeros(orig_size_xyz)

    for edge in input_edges:
        node1, node2 = edge
        new_state += calc_node_state(network, orig_size_xyz, node1) * \
            network.graph.edges[node1,node2]["weight"]

    network.graph.nodes[node_name]["state"] = new_state

    return network.graph.nodes[node_name]["function"](new_state)


def mutate_network(network):
    mut_func_args = inspect.getargspec(network.mutate)
    mut_func_args = [0 for _ in range(1, len(mut_func_args.args))]
    choice = random.choice(range(len(mut_func_args)))
    mut_func_args[choice] = 1
    variation_type, variation_degree = network.mutate(*mut_func_args)


def natural_sort(l, reverse):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)

def make_one_shape_only(output_state, mask=None):
    """Find the largest continuous arrangement of True elements after applying boolean mask.

    Avoids multiple disconnected softbots in simulation counted as a single individual.

    Parameters
    ----------
    output_state : numpy.ndarray
        Network output

    mask : bool mask
        Threshold function applied to output_state

    Returns
    -------
    part_of_ind : bool
        True if component of individual

    """
    if mask is None:
        def mask(u): return np.greater(u, 0)

    # print output_state
    # sys.exit(0)

    one_shape = np.zeros(output_state.shape, dtype=np.int32)

    if np.sum(mask(output_state)) < 2:
        one_shape[np.where(mask(output_state))] = 1
        return one_shape

    else:
        not_yet_checked = []
        for x in range(output_state.shape[0]):
            for y in range(output_state.shape[1]):
                for z in range(output_state.shape[2]):
                    not_yet_checked.append((x, y, z))

        largest_shape = []
        queue_to_check = []
        while len(not_yet_checked) > len(largest_shape):
            queue_to_check.append(not_yet_checked.pop(0))
            this_shape = []
            if mask(output_state[queue_to_check[0]]):
                this_shape.append(queue_to_check[0])

            while len(queue_to_check) > 0:
                this_voxel = queue_to_check.pop(0)
                x = this_voxel[0]
                y = this_voxel[1]
                z = this_voxel[2]
                for neighbor in [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]:
                    if neighbor in not_yet_checked:
                        not_yet_checked.remove(neighbor)
                        if mask(output_state[neighbor]):
                            queue_to_check.append(neighbor)
                            this_shape.append(neighbor)

            if len(this_shape) > len(largest_shape):
                largest_shape = this_shape

        for loc in largest_shape:
            one_shape[loc] = 1

        return one_shape

class OrderedGraph(DiGraph):
    """Create a graph object that tracks the order nodes and their neighbors are added."""
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict

class Network(object):
    """Base class for networks."""

    input_node_names = []

    def __init__(self, output_node_names):
        self.output_node_names = output_node_names
        self.graph = OrderedGraph()  # preserving order is necessary for checkpointing
        self.freeze = False
        self.allow_neutral_mutations = False
        self.num_consecutive_mutations = 1 # ALICAN: TODO: bunu belki hardcoded yapmayabiliriz
        self.direct_encoding = False

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def set_input_node_states(self, *args, **kwargs):
        raise NotImplementedError

    def mutate(self, *args, **kwargs):
        raise NotImplementedError

class CPPN(Network):
    """A Compositional Pattern Producing Network"""

    input_node_names = ['x', 'y', 'z', 'd', 'b']
    activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs]

    def __init__(self, output_node_names):
        Network.__init__(self, output_node_names)
        self.set_minimal_graph()
        #self.mutate()

    def set_minimal_graph(self):
        """Create a simple graph with each input attached to each output"""
        for name in self.input_node_names:
            self.graph.add_node(name, type="input", function=None)

        for name in self.output_node_names:
            self.graph.add_node(name, type="output", function=sigmoid)

        for input_node in nx.nodes(self.graph):
            if self.graph.nodes[input_node]["type"] == "input":
                for output_node in nx.nodes(self.graph):
                    if self.graph.nodes[output_node]["type"] == "output":
                        self.graph.add_edge(input_node, output_node, weight=0.0)

    def set_input_node_states(self, orig_size_xyz):
        input_x = np.zeros(orig_size_xyz)
        input_y = np.zeros(orig_size_xyz)
        input_z = np.zeros(orig_size_xyz)
        for x in range(orig_size_xyz[0]):
            for y in range(orig_size_xyz[1]):
                for z in range(orig_size_xyz[2]):
                    input_x[x, y, z] = x
                    input_y[x, y, z] = y
                    input_z[x, y, z] = z

        input_x = normalize(input_x)
        input_y = normalize(input_y)
        input_z = normalize(input_z)
        input_d = normalize(np.power(np.power(input_x, 2) + np.power(input_y, 2) + np.power(input_z, 2), 0.5))
        input_b = np.ones(orig_size_xyz)  # TODO: check input_d (above): changed pow -> np.power without testing

        for name in self.graph.nodes():
            if name == "x":
                self.graph.nodes[name]["state"] = input_x
                self.graph.nodes[name]["evaluated"] = True
            if name == "y":
                self.graph.nodes[name]["state"] = input_y
                self.graph.nodes[name]["evaluated"] = True
            if name == "z":
                self.graph.nodes[name]["state"] = input_z
                self.graph.nodes[name]["evaluated"] = True
            if name == "d":
                self.graph.nodes[name]["state"] = input_d
                self.graph.nodes[name]["evaluated"] = True
            if name == "b":
                self.graph.nodes[name]["state"] = input_b
                self.graph.nodes[name]["evaluated"] = True

    def mutate(self, num_random_node_adds=5, num_random_node_removals=0, num_random_link_adds=10,
               num_random_link_removals=5, num_random_activation_functions=100, num_random_weight_changes=100):

        # TODO: set default arg val via brute force search
        # TODO: weight std is defaulted to 0.5, to change this we can't just put it in args of mutate() b/c getargspec
        # is used in create_new_children_through_mutation() to automatically pick the mutation type.

        variation_degree = None
        variation_type = None

        for _ in range(num_random_node_adds):
            variation_degree = self.add_node()
            variation_type = "add_node"

        for _ in range(num_random_node_removals):
            variation_degree = self.remove_node()
            variation_type = "remove_node"

        for _ in range(num_random_link_adds):
            variation_degree = self.add_link()
            variation_type = "add_link"

        for _ in range(num_random_link_removals):
            variation_degree = self.remove_link()
            variation_type = "remove_link"

        for _ in range(num_random_activation_functions):
            variation_degree = self.mutate_function()
            variation_type = "mutate_function"

        for _ in range(num_random_weight_changes):
            variation_degree = self.mutate_weight()
            variation_type = "mutate_weight"

        self.prune_network()
        return variation_type, variation_degree

    ###############################################
    #   Mutation functions
    ###############################################

    def add_node(self):
        # choose two random nodes (between which a link could exist)
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(list(self.graph.edges()))
        node1 = this_edge[0]
        node2 = this_edge[1]

        # create a new node hanging from the previous output node
        new_node_index = self.get_max_hidden_node_index()
        self.graph.add_node(new_node_index, type="hidden", function=random.choice(self.activation_functions))
        # random activation function here to solve the problem with admissible mutations in the first generations
        self.graph.add_edge(new_node_index, node2, weight=1.0)

        # if this edge already existed here, remove it
        # but use it's weight to minimize disruption when connecting to the previous input node
        if (node1, node2) in nx.edges(self.graph):
            weight = self.graph.edges[node1,node2]["weight"]
            self.graph.remove_edge(node1, node2)
            self. graph.add_edge(node1, new_node_index, weight=weight)
        else:
            self.graph.add_edge(node1, new_node_index, weight=1.0)
            # weight 0.0 would minimize disruption of new edge
            # but weight 1.0 should help in finding admissible mutations in the first generations
        return ""

    def remove_node(self):
        hidden_nodes = list(set(self.graph.nodes()) - set(self.input_node_names) - set(self.output_node_names))
        if len(hidden_nodes) == 0:
            return "NoHiddenNodes"
        this_node = random.choice(hidden_nodes)

        # if there are edge paths going through this node, keep them connected to minimize disruption
        incoming_edges = self.graph.in_edges(nbunch=[this_node])
        outgoing_edges = self.graph.out_edges(nbunch=[this_node])

        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                w = self.graph.edges[incoming_edge[0],this_node]["weight"] * \
                    self.graph.edges[this_node,outgoing_edge[1]]["weight"]
                self.graph.add_edge(incoming_edge[0], outgoing_edge[1], weight=w)

        self.graph.remove_node(this_node)
        return ""

    def add_link(self):
        done = False
        attempt = 0
        while not done:
            done = True

            # choose two random nodes (between which a link could exist, *but doesn't*)
            node1 = random.choice(list(self.graph.nodes()))
            node2 = random.choice(list(self.graph.nodes()))
            while (not self.new_edge_is_valid(node1, node2)) and attempt < 999:
                node1 = random.choice(list(self.graph.nodes()))
                node2 = random.choice(list(self.graph.nodes()))
                attempt += 1
            if attempt > 999:  # no valid edges to add found in 1000 attempts
                done = True

            # create a link between them
            if random.random() > 0.5:
                self.graph.add_edge(node1, node2, weight=0.1)
            else:
                self.graph.add_edge(node1, node2, weight=-0.1)

            # If the link creates a cyclic graph, erase it and try again
            if self.has_cycles():
                self.graph.remove_edge(node1, node2)
                done = False
                attempt += 1
            if attempt > 999:
                done = True
        return ""

    def remove_link(self):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_link = random.choice(list(self.graph.edges()))
        self.graph.remove_edge(this_link[0], this_link[1])
        return ""

    def mutate_function(self):
        this_node = random.choice(list(self.graph.nodes()))
        while this_node in self.input_node_names:
            this_node = random.choice(list(self.graph.nodes()))
        old_function = self.graph.nodes[this_node]["function"]
        while self.graph.nodes[this_node]["function"] == old_function:
            self.graph.nodes[this_node]["function"] = random.choice(self.activation_functions)
        return old_function.__name__ + "-to-" + self.graph.nodes[this_node]["function"].__name__

    def mutate_weight(self, mutation_std=0.5):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(list(self.graph.edges()))
        node1 = this_edge[0]
        node2 = this_edge[1]
        old_weight = self.graph[node1][node2]["weight"]
        new_weight = old_weight
        while old_weight == new_weight:
            new_weight = random.gauss(old_weight, mutation_std)
            new_weight = max(-1.0, min(new_weight, 1.0))
        self.graph[node1][node2]["weight"] = new_weight
        return float(new_weight - old_weight)

    ###############################################
    #   Helper functions for mutation
    ###############################################

    def prune_network(self):
        """Remove erroneous nodes and edges post mutation."""
        done = False
        while not done:
            done = True
            for node in list(self.graph.nodes()):
                if len(self.graph.in_edges(nbunch=[node])) == 0 and \
                                node not in self.input_node_names and \
                                node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

            for node in list(self.graph.nodes()):
                if len(self.graph.out_edges(nbunch=[node])) == 0 and \
                                node not in self.input_node_names and \
                                node not in self.output_node_names:
                    self.graph.remove_node(node)
                    done = False

    def has_cycles(self):
        """Return True if the graph contains simple cycles (elementary circuits).

        A simple cycle is a closed path where no node appears twice, except that the first and last node are the same.

        """
        return sum(1 for _ in nx.simple_cycles(self.graph)) != 0

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.nodes[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        if node1 == node2:
            return False
        if self.graph.nodes[node1]['type'] == "output":
            return False
        if self.graph.nodes[node2]['type'] == "input":
            return False
        if (node2, node1) in nx.edges(self.graph):
            return False
        if (node1, node2) in nx.edges(self.graph):
            return False
        return True


