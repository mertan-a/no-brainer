import numpy as np

from cppn import mutate_network, calc_outputs, CPPN
from evogym import is_connected

class BODY(object):
    def __init__(self, type):
        self.type = type

    def mutate(self):
        raise NotImplementedError

    def to_phenotype(self):
        raise NotImplementedError

    def is_valid(self):
        raise NotImplementedError

class CPPN_BODY(BODY):
    def __init__(self, args):
        BODY.__init__(self, "cppn")
        self.args = args
        if self.args.task == 'BasicEnv-v0':
            self.output_node_names = ["0", "1", "2", "3", "4", "5", "6"]
        elif self.args.task == 'BasicDirectionalEnv-v0':
            self.output_node_names = ["0", "1", "2", "3", "4", "5", "6"]
        elif self.args.task == 'LogicEnv-v0':
            self.output_node_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        else:
            raise NotImplementedError
        self.network = CPPN(output_node_names=self.output_node_names)
        self.orig_size_xyz = [1,self.args.bounding_box[0],self.args.bounding_box[1]]
        self.mutate()

    def to_phenotype(self):
        calc_outputs(
            self.network, self.orig_size_xyz, self)
        voxel_data = np.stack(
            [self.network.graph.nodes[node_name]["state"] for node_name in self.output_node_names], axis=-1)
        voxel_data = np.squeeze(voxel_data)
        voxel_data = np.argmax(voxel_data, axis=-1)
        return voxel_data

    def mutate(self):
        is_valid = False
        while not is_valid:
            mutate_network(self.network)
            is_valid = self.is_valid()

    def is_valid(self, min_percent_full=0.3):
        voxel_data = self.to_phenotype()
        if np.isnan(voxel_data).any():
            return False
        if np.sum(voxel_data == 3) + np.sum(voxel_data == 4) < 3:
            return False
        if is_connected(voxel_data) == False:
            return False
        return True

    def count_existing_voxels(self):
        voxel_data = self.to_phenotype()
        return np.sum(voxel_data > 0)

    def count_active_voxels(self):
        voxel_data = self.to_phenotype()
        return np.sum(voxel_data == 3) + np.sum(voxel_data == 4)

    def count_perception_voxels(self):
        voxel_data = self.to_phenotype()
        return np.sum(voxel_data == 5) + np.sum(voxel_data == 6)

if __name__ == "__main__":
    body = CPPN_BODY(bounding_box=(1, 10, 10,))
    print(body.network.graph)
    #import networkx as nx
    #import matplotlib.pyplot as plt
    #nx.draw(body.network.graph, with_labels=True)
    #plt.show()
    while True:
        body.mutate()
        print(body.network.graph)
        print(body.is_valid())
        phenotyped_body = body.to_phenotype()
        print(phenotyped_body)
        # check the number of unique materials
        if len(np.unique(phenotyped_body)) < 7:
            continue
        input()




