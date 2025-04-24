# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from PathGraph import PathGraph
from PathGraph import FFT
from tqdm import tqdm
# ------------------------------------------------------------

signal_size = 2560
root = r'D:\data\PU'
dataname= {0:["K004_09_07_10.csv","KA04_09_07_10.csv", "KA16_09_07_10.csv", "KI21_09_07_10.csv", "KI18_09_07_10.csv", "KB24_09_07_10.csv"],
            1:["K004_15_01_10.csv","KA04_15_01_10.csv", "KA16_15_01_10.csv", "KI21_15_01_10.csv", "KI18_15_01_10.csv", "KB24_15_01_10.csv"],
            2:["K004_15_07_04.csv","KA04_15_07_04.csv", "KA16_15_07_04.csv", "KI21_15_07_04.csv", "KI18_15_07_04.csv", "KB24_15_07_04.csv"],
            3:["K004_15_07_10.csv","KA04_15_07_10.csv", "KA16_15_07_10.csv", "KI21_15_07_10.csv", "KI18_15_07_10.csv", "KB24_15_07_10.csv"]}
label = [0,1,2,3,4,5]
label2 = [0,1,2,3,4,5]
label3 = [0,1,2,3,4,5]
label4 = [0,1,2,3,4,5]

# generate Training Dataset and Testing Dataset
def get_files(root, N):
    data = []
    for i in tqdm(range(len(dataname[N[0]]))):
        path1 = os.path.join(root,dataname[N[0]][i])
        data1 = data_load(path1, label=label[i])
        data += data1
    return data

def data_load(root, label):
    fl = []
    j=0
    pdata = pd.read_csv(root, sep='\t', usecols=[j], header=None, )
    pdata = pdata.values
    pdata = (pdata - pdata.min()) / (pdata.max() - pdata.min())
    pdata = pdata.reshape(-1,)
    fl.append(pdata)
    fl = np.array(fl)

    data = []
    start, end = 0, signal_size
    while end <= fl[:,64000:signal_size * 200 + 64000].shape[1]:
        x = fl[0,start:end]
        x = FFT(x)
        data.append(x)
        start += signal_size
        end += signal_size
    graphset = PathGraph(5,data,label)
    return graphset
    

def PUPath(domain,batch_size):
    '''
    This function is used to generate the final training set and test set.
    '''
    if domain == "client_1":
        list_data = get_files(root, [0])
    elif domain == "client_2":
        list_data = get_files(root, [1])
    elif domain == "client_3":
        list_data = get_files(root, [2])
    elif domain == "client_4":
        list_data = get_files(root, [3])
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))

    train_dataset, val_dataset = train_test_split(list_data, test_size=0.5, random_state=40)
    train_loader=  DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                      )   
    val_loader=  DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                      )                     
    return train_loader, val_loader



