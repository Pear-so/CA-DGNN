import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data
from scipy.spatial.distance import pdist

###Dynamic edge feature
def update_edge_features(node_features, batch_size, num_nodes_per_graph):

    new_weights = torch.empty(0, dtype=torch.float)
    for batch_idx in range(batch_size // num_nodes_per_graph):
        start_idx = batch_idx * num_nodes_per_graph
        end_idx = (batch_idx + 1) * num_nodes_per_graph
        graph_node_features = node_features[start_idx:end_idx]
        distance = []

        for i in range(num_nodes_per_graph - 1):
            v_1 = graph_node_features[i].unsqueeze(0)
            v_2 = graph_node_features[i + 1].unsqueeze(0)
            likely = torch.cdist(v_1, v_2, p=2)
            distance.append(likely.item())

        beata = torch.tensor(distance).mean()
        new_weight = torch.exp(-torch.tensor(distance) ** 2 / (2 * beata ** 2))
        new_weights = torch.cat((new_weights, new_weight), dim=0)
    return new_weights.to(node_features.device)

###PathGraph
    '''
    This function is used to generate the graph via PathGraph method.
    Acknowledgments: 
        The code of PathGraph is improved based on the original work by Dr. Tianfu Li from Xi'an Jiaotong University. We appreciate his open-source spirit.
    Reference: 
        Li, T. et al., "A survey on graph-based methods for time series analysis", Mech. Syst. Signal Process. 159 (2021) 108653. 
        https://doi.org/10.1016/j.ymssp.2021.108653
    Code Implementation Reference: 
        PHM-GNN Benchmark repository by Dr. Tianfu Li: 
        https://github.com/HazeDT/PHMGNNBenchmark    
    '''
def PathGraph(interval,data,label):
    a, b = 0, interval
    graph_list = []
    while b <= len(data):
        graph_list.append(data[a:b])
        a += interval
        b += interval
    graphset = Gen_graph("PathGraph",graph_list,label)
    return graphset

def Gen_graph(graphType, data, label):
    data_list = []
    if graphType == 'PathGraph':
        for i in range(len(data)):

            graph_feature = data[i]
            labels = [label]
            node_edge, w = Path_attr(graph_feature)
            node_features = torch.tensor(graph_feature, dtype=torch.float)
            graph_label = torch.tensor(labels, dtype=torch.long)
            edge_index = torch.tensor(node_edge, dtype=torch.long)
            edge_features = torch.tensor(w, dtype=torch.float)
            graph = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_features)
            data_list.append(graph)
    else:
        print("This GraphType is not included!")
    return data_list

def Path_attr(data):
    node_edge = [[], []]
    for i in range(len(data) - 1):
        node_edge[0].append(i)
        node_edge[1].append(i + 1)
    distance = []
    for j in range(len(data) - 1):
        v_1 = data[j]
        v_2 = data[j + 1]
        combine = np.vstack([v_1, v_2])
        likely = pdist(combine, 'euclidean')
        distance.append(likely[0])
    beata = np.mean(distance)
    w = np.exp((-(np.array(distance)) ** 2) / (2 * (beata ** 2)))  #Gussion kernel高斯核

    return node_edge,w

###FFT
def FFT(x):
    x = np.fft.fft(x)
    x = np.abs(x) / len(x)
    x = x[range(int(x.shape[0] / 2))]
    return x
