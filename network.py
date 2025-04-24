# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import warnings
from torch_geometric.nn import GraphConv,  BatchNorm # noqa
from PathGraph import update_edge_features

###Fully Connected Layers
class FC_Layers(nn.Module):
    def __init__(self, input_dim, fc_dim_1,num_cls):
        super(FC_Layers, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, fc_dim_1),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_dim_1, num_cls),
        )
        self.dropout = nn.Dropout(0.6)
    def forward(self, x):
        z_1 = self.fc1(x)
        z_2 = self.dropout(z_1)
        y = self.fc2(z_2)
        p = F.softmax(y, dim=1)
        return y, z_2, p
    
###DGNN
class DGNN(torch.nn.Module):
    def __init__(self, feature,pooltype):
        super(DGNN, self).__init__()
        self.GConv1 = GraphConv(feature,512)
        self.bn1 = BatchNorm(512)
        self.GConv2 = GraphConv(512,512)
        self.bn2 = BatchNorm(512)
    def forward(self, x, edge_index, edge_weight, batch, pooltype):        
        x = self.GConv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x1 = global_mean_pool(x, batch)
        edge_weight = update_edge_features(x, batch.size(0), 5)
        x = self.GConv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x2 = global_mean_pool(x, batch)
        x = x2+x1  
        return x