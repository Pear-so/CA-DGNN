# -*- coding: utf-8 -*-

import network
import numpy as np
import scipy.io as io
import torch
from datetime import datetime
from torch.autograd import Variable
import CMMD
import sys
import os
import argparse
import csv
import torch.nn as nn

from PUPath_data import PUPath

##################################
# Set Parameter
##################################
# Your own path
data_folder = './dataset' # path of dataset
save_folder = './results' # path for saving results
log_folder = './logs' # path for saving logs
model_dir = './pre_model/' # path of pretrained model

#################################################################
#
#################################################################
'''
sub_dir = 'CA_GD_12' + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
save_dir = os.path.join('./model_save/PU', sub_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
'''
################################################################
class MyModel(nn.Module):
    def __init__(self, DNN, FC):
        super(MyModel, self).__init__()
        self.DNN = DNN
        self.FC = FC
    def forward(self, x, edge_index, edge_weight, batch, pooltype):
        x = self.DNN(x, edge_index, edge_weight, batch, pooltype)
        y, y, p = self.FC(x)
        return y
##################################
# Experiment Main Function
##################################

# target_column_index = 0
taracc=[]
Loss=[]
def Experiment_Main(config):#主函数Experiment_Main
    # Parameter
    FC_dim_1 = int(config['FC_dim_1'])
    FC_dim_2 = int(config['FC_dim_2'])
    exp_times = int(config['exp_times'])
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    lr = float(config['lr'])
    optim_param = config['optim_param']
    pooltype = config['pooltype']#默认为edgepool

    # Prepare Data

    source_tr_loader,source_te_loader = PUPath("client_2", batch_size=batch_size)
    target_tr_loader,target_te_loader = PUPath("client_1", batch_size=batch_size)
       
    for exp_iter in range(exp_times):


            DNN = network.HoGCN(feature=1280, pooltype=pooltype)
            FC_input_dim = 512
            num_cls = 6
            FC = network.FC_Layers(FC_input_dim,FC_dim_1,num_cls)
            FC.apply(CMMD.weights_init)

            FC.cuda()
            DNN.cuda()

            
            # Optimizer
            if optim_param == 'GD':
                optimizer_dict = [{"params": filter(lambda p: p.requires_grad, DNN.parameters()), "lr": 1},
                          {"params": filter(lambda p: p.requires_grad, FC.parameters()), "lr": 2}]
                optimizer = torch.optim.SGD(optimizer_dict, lr=lr, momentum=0.9, weight_decay=0.005, nesterov=True)
                param_lr = []
                for param_group in optimizer.param_groups:
                    param_lr.append(param_group["lr"])
            elif optim_param == 'Adam':
                beta1=0.9
                beta2=0.999
                optimizer = torch.optim.Adam([{'params':DNN.parameters(), 'lr': lr*0.1},
                                        {'params':FC.parameters()}], lr*1.5,
                                        [beta1, beta2], weight_decay=0.01)
            else:
                sys.exit('Error: invalid optimizer')
            
            ####################
            # Training 
            ####################
            iter_optim = 1
            for step in range(epochs):
                epoch_time_start = datetime.now()
                ####################
                # Mini-batch training 
                ####################
                for data_s , data_t in zip(source_tr_loader,target_tr_loader):
                    
                # switch to training mode
                    DNN.train()
                    FC.train()
                    
                    if optim_param == 'GD':
                    # upadate lr
                        optimizer = CMMD.inv_lr_scheduler(param_lr, optimizer, iter_optim, init_lr=lr, gamma=0.01, power=0.75,
                                          weight_decay=0.0005)
                        iter_optim += 1

                    # load data
                    X_s, edgeIndex_s, lab_s, batch_s ,edgeWeight_s = data_s.x, data_s.edge_index, data_s.y, data_s.batch, data_s.edge_attr##
                    X_t, edgeIndex_t, lab_t, batch_t ,edgeWeight_t = data_t.x, data_t.edge_index, data_t.y, data_t.batch, data_t.edge_attr##

                    X_s, lab_s, edgeIndex_s, batch_s ,edgeWeight_s= CMMD.to_var(X_s), CMMD.to_var(lab_s), CMMD.to_var(edgeIndex_s), CMMD.to_var(batch_s), CMMD.to_var(edgeWeight_s)
                    X_t, lab_t, edgeIndex_t, batch_t ,edgeWeight_t= CMMD.to_var(X_t), CMMD.to_var(lab_t), CMMD.to_var(edgeIndex_t), CMMD.to_var(batch_t), CMMD.to_var(edgeWeight_t)



                    onehot_s = torch.zeros(lab_s.shape[0], num_cls).cuda().scatter_(1, lab_s.view(-1,1), 1)
                    onehot_s = Variable(onehot_s).cuda()
                    
                    # Init gradients
                    DNN.zero_grad()
                    FC.zero_grad()
                    # Forward propagate
                    _, Z_s, prob_s = FC(DNN(X_s, edgeIndex_s, edgeWeight_s, batch_s, pooltype))
                    _, Z_t, prob_t = FC(DNN(X_t, edgeIndex_t, edgeWeight_t, batch_t, pooltype))
                    
                    # plab_t = prob_t.detach().max(1)[1]
                    ####################
                    # Loss Objective
                    ####################
                    # Cross-Entropy
                    CE_loss = CMMD.Cross_Entropy(prob_s, lab_s)

                    ####
                    H_term  = CMMD.Entropy(prob_t)
                    mean_prob_t = prob_t.mean(dim=0, keepdim=True)
                    H_mean_term = CMMD.Entropy(mean_prob_t)
                    Tar_Ent_lambda=0.05
                    Tar_Ent_loss=Tar_Ent_lambda*(H_term - H_mean_term)
                    L_s = CMMD.kernel_gram_KLN(onehot_s,onehot_s,lab_s.shape[0],num_cls)
                    L_t = CMMD.kernel_gram_KLN(prob_t.detach(),prob_t.detach(),lab_t.shape[0],num_cls)
                    L_st = CMMD.kernel_gram_KLN(onehot_s,prob_t.detach(),lab_s.shape[0],num_cls)
                    K_s = CMMD.kernel_gram_z_KLN(Z_s,Z_s,lab_s.shape[0],Z_s.shape[1])
                    K_t = CMMD.kernel_gram_z_KLN(Z_t,Z_t,lab_s.shape[0],Z_s.shape[1])
                    K_st = CMMD.kernel_gram_z_KLN(Z_t,Z_s,lab_s.shape[0],Z_s.shape[1])
                    I = torch.eye(lab_s.shape[0])
                    I = Variable(I).cuda()
                    lambda_ = 0.2
                    Inv_K_s = torch.inverse(K_s + lambda_*I)
                    Inv_K_t = torch.inverse(K_t + lambda_*I)
                    cmmd_t = lambda_ * (torch.trace(torch.mm(torch.mm(torch.mm(K_s, Inv_K_s), L_s), Inv_K_s)) +\
                                 torch.trace(torch.mm(torch.mm(torch.mm(K_t, Inv_K_t), L_t),Inv_K_t))- \
                                    2 * torch.trace(torch.mm(torch.mm(torch.mm(K_st, Inv_K_s) ,L_st ), Inv_K_t)))
                    cmmd_weight =1
                    CMMD_loss = cmmd_t * cmmd_weight
 
                    O_loss = CE_loss + Tar_Ent_loss + CMMD_loss
                    
                    # Backward propagate
                    DNN.zero_grad()
                    FC.zero_grad()
                    O_loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    
                ####################
                # Testing 
                ####################
                # switch to testing mode
                DNN.eval()
                FC.eval()
                # evaluate Model
                source_acc = CMMD.classification_accuracy(source_te_loader,DNN,FC,pooltype)*100
                target_acc = CMMD.classification_accuracy(target_te_loader,DNN,FC,pooltype)*100
                '''
                if step > 280:
                    model_save = MyModel(DNN, FC)
                    model_state_dic = model_save.state_dict()
                    torch.save(model_state_dic,
                               os.path.join(save_dir, '{}-{:.4f}-new_model.pth'.format(step, target_acc)))
                '''
                ####################
                # Report results
                ####################
                # time
                epoch_time_end = datetime.now()
                seconds = (epoch_time_end - epoch_time_start).seconds
                minutes = seconds//60
                second = seconds%60
                hours = minutes//60
                minute = minutes%60
                
                taracc.append(target_acc)
                # print result
                # print('====================== [%1s] %1s→%1s: Experiment %1s Epoch %1s ================='%(dataset,source_domain,target_domain,exp_iter+1,step+1))
                print('Epoch :{}.'.format(step))
                print('Source Accuracy: %1s'%source_acc)
                print('Target Accuracy: %1s'%target_acc)
                print('Cross-Entropy Loss: %1s'%(CE_loss.data.data.cpu().numpy()))#将cuda转化为cpu
                print('Target Entropy Loss: %1s'%(Tar_Ent_loss.data.cpu().numpy()))#将cuda转化为cpu
                print('CMMD: %1s'%(CMMD_loss.data.cpu().numpy()))#将cuda转化为cpu
                # print('KB Loss: %1s'%(KB_loss.data.cpu().numpy()))
                
                print('Overall Loss: %1s'%(O_loss.data.cpu().numpy()))#将cuda转化为cpu
                print('Current epoch [train & test] time cost: %1s Hour %1s Minutes %1s Seconds'%(hours,minute,second))

                # empty network cache
                torch.cuda.empty_cache()

target_column_index=15
    ####################
    # Save results
    ####################


if __name__ == "__main__":#
    parser = argparse.ArgumentParser(description='CMMD')#
    parser.add_argument('--dataset', type=str, default='RefurbishedOffice31', choices=['ImageCLEF', 'OfficeHome', 'Office10', 'RefurbishedOffice31'])
    parser.add_argument('--net', type=str, default='ResNet-50', choices=['ResNet-50', 'AlexNet'])#

    parser.add_argument('--FC_dim_1', type=str, default='256', help="dimension of the 1st FC layer")#
    parser.add_argument('--FC_dim_2', type=str, default='32', help="dimension of the 2nd FC layer")
    parser.add_argument('--exp_times', type=str, default='1', help="numbers of random experiment")
    parser.add_argument('--epochs', type=str, default='300', help="maximum training epochs")
    parser.add_argument('--batch_size', type=str, default='64', help="training batch_size; 40 for ResNet-50 and 128 for AlexNet")
    parser.add_argument('--lr', type=str, default='0.001', help="learning rate")
    parser.add_argument('--optim_param', type=str, default='GD', choices=['Adam', 'GD'])
    parser.add_argument('--pooltype', type=str, default='EdgePool', choices=['TopKPool', 'EdgePool', 'ASAPool', 'SAGPool'], help='For the Graph classification task')
    ####
    parser.add_argument('--Tar_Ent_epoch', type=str, default='1', help="training with target entropy loss after # epochs")
    parser.add_argument('--CKB_epoch', type=str, default='1', help="training with CKB loss after # epochs")
    parser.add_argument('--inv_epsilon', type=str, default='1e-3', help="regularization parameter of kernel matrix inverse")
    parser.add_argument('--GPU_device', type=str, nargs='?', default='0', help="set GPU device for training")
    parser.add_argument('--seed', type=str, default='0', help="random seed")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    
    config = {}#
    config['dataset'] = args.dataset#
    config['net'] = args.net
    config['FC_dim_1'] = args.FC_dim_1
    config['FC_dim_2'] = args.FC_dim_2
    config['exp_times'] = args.exp_times
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['Tar_Ent_epoch'] = args.Tar_Ent_epoch
    config['CKB_epoch'] = args.CKB_epoch
    config['inv_epsilon'] = args.inv_epsilon
    config['lr'] = args.lr
    config['optim_param'] = args.optim_param
    config['GPU_device'] = args.GPU_device
    config['pooltype'] = args.pooltype

    ##################################
    # Random Seeds
    ##################################
    torch.manual_seed(int(args.seed)) 

    Experiment_Main(config)#

