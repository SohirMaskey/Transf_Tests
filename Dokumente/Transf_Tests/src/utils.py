## Standard libraries
import numpy as np


## PyTorch
import torch


from torch_geometric.data import Data


import networkx as nx
from torch_geometric.utils import from_networkx


#for sampling random graphs
from torch_cluster import radius_graph
from torch_geometric.utils import erdos_renyi_graph, stochastic_blockmodel_graph, barabasi_albert_graph



def SBM_Gen(N, blocks, edge_prob_block, edge_prob_all=0):
    """
    Generates a SBM graph in [0,1] by first sampling random uniformly N nodes. Then, seperating
    them according to blocks, block are floats seperting [0,1], i.e., list of length number_of_blocks
    edge_prob_block: list of length number_of_blocks: every element give the edge_probability in it.
    edge_prob_all: integer giving the interblock probabiltiy of an edge.
    """
    s = np.random.uniform(0, 1, N)
    intervals = blocks
    sizes = []
    for i in range(0, len(intervals)):
        size = 0
        for j in range(0, len(s)):
            if i == 0:
                if s[j] > 0 and s[j] <= intervals[0]:
                    size = size + 1
            elif s[j] > intervals[i - 1] and s[j] <= intervals[i]:
                size = size + 1
        sizes.append(size)

    ss = torch.tensor(s).reshape(N, 1)
    probsPerInt = edge_prob_block
    y = edge_prob_all

    probs = []
    for i in range(0, len(probsPerInt)):
        prob = [y] * len(probsPerInt)
        prob[i] = probsPerInt[i]
        probs.append(prob)

    return nx.stochastic_block_model(sizes, probs, seed=0)


def make_datalist(sz, gr_number):
    """
    Input: - sz: int, size of the output graphs
           - gr_number: int, size of the output data set

    Generates gr_number of graph of size for 8 different graphon model.

    Output: A list data_list of size 8*gr_number. Each element is a Data-Graph with sz nodes. Generated from 8 different graphon models
    """

    dl = []

    radius = 0.1

    for i in range(0, gr_number):
        pos = torch.rand(sz, 2)

        batch = torch.zeros(int(sz)).type(torch.LongTensor)
        # x = low_pass(pos).reshape(50,1)
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = radius_graph(pos, r=radius, batch=batch, loop=False, max_num_neighbors=sz)
        gr = Data(x=x, edge_index=edge_index, y=torch.ones(1).to(torch.long))

        dl.append(gr)
    radius = 0.5

    for i in range(0, gr_number):
        pos = torch.rand(sz, 2)

        batch = torch.zeros(int(sz)).type(torch.LongTensor)
        # x = low_pass(pos).reshape(50,1)
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = radius_graph(pos, r=radius, batch=batch, loop=False, max_num_neighbors=(sz))
        gr = Data(x=x, edge_index=edge_index, y=torch.zeros(1).to(torch.long))

        dl.append(gr)
    p = 0.1

    for i in range(0, gr_number):
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = erdos_renyi_graph(sz, p)
        gr = Data(x=x, edge_index=edge_index, y=2 * torch.ones(1).to(torch.long))

        dl.append(gr)
    p = 0.5

    for i in range(0, gr_number):
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = erdos_renyi_graph(sz, p)
        gr = Data(x=x, edge_index=edge_index, y=3 * torch.ones(1).to(torch.long))

        dl.append(gr)

    blocks = [1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1]
    edge_prob_block = [4 / 8, 1 / 8, 1 / 32, 1 / 32, 1 / 16, 1 / 3, 1 / 8, 1 / 64]

    for i in range(0, gr_number):
        g = SBM_Gen(sz, blocks, edge_prob_block, edge_prob_all=1 / 4)
        g = from_networkx(g)
        g.y = 4 * torch.ones(1).to(torch.long)
        g.x = torch.ones(g.num_nodes).reshape(g.num_nodes, 1)
        del g.num_nodes
        del g.block
        dl.append(g)

    blocks = [1 / 4, 2 / 4, 3 / 4, 1]
    edge_prob_block = [6 / 16, 1 / 16, 2 / 16, 3 / 16]

    for i in range(0, gr_number):
        g = SBM_Gen(sz, blocks, edge_prob_block, edge_prob_all=0)
        g = from_networkx(g)
        g.y = 5 * torch.ones(1).to(torch.long)
        g.x = torch.ones(g.num_nodes).reshape(g.num_nodes, 1)
        del g.num_nodes
        del g.block
        dl.append(g)

    p = 5

    for i in range(0, gr_number):
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = barabasi_albert_graph(sz, p)
        gr = Data(x=x, edge_index=edge_index, y=6 * torch.ones(1).to(torch.long))

        dl.append(gr)

    p = 10

    for i in range(0, gr_number):
        x = torch.ones(sz).reshape(sz, 1)
        edge_index = barabasi_albert_graph(sz, p)
        gr = Data(x=x, edge_index=edge_index, y=7 * torch.ones(1).to(torch.long))

        dl.append(gr)
    return dl