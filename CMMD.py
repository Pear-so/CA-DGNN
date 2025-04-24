# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

###
def kernel_gram_KLN(x,y,n,d):
    zx = x.repeat(1,n).resize(n*n,d)
    zy = y.repeat(n,1)
    z = torch.pow(torch.nn.functional.pairwise_distance(zx,zy),2)
    zz = z.resize(n,n)
    res = torch.zeros(n,n)
    res = Variable(res).cuda()
    bw=1
    res = torch.exp(-zz/(2*bw))
    return res

###
def kernel_gram_z_KLN(x,y,n,d):
    mn = 5
    bw = [1,5,10,20,40]
    zx = x.repeat(1,n).resize(n*n,d)
    zy = y.repeat(n,1)
    z = torch.pow(torch.nn.functional.pairwise_distance(zx,zy),2)
    zz = z.resize(n,n)
    res = torch.zeros(n,n)
    res = Variable(res).cuda()
    for i in range(mn):
        res = res + torch.exp(-zz/(2*bw[i]))
    return res/5

###
def weights_init(m):
    """Initialize network parameters."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)         
        
###
def to_var(x):
    """Convert numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

###
def classification_accuracy(data_loader,DNN,FC,pooltype):
    with torch.no_grad():
        correct = 0
        for batch_idx, data_te in enumerate(data_loader):
            X, edgeIndex, lab, batch ,edgeWeight= data_te.x, data_te.edge_index, data_te.y, data_te.batch, data_te.edge_attr##

            X, lab, edgeIndex, batch,edgeWeight  = to_var(X), to_var(lab).long().squeeze(), to_var(edgeIndex), to_var(batch), to_var(edgeWeight)
            _, _,prob = FC(DNN(X, edgeIndex,edgeWeight, batch, pooltype))
            plab = prob.data.max(1)[1]
            correct += plab.eq(lab.data).cpu().sum()
        accuracy = correct.item() / len(data_loader.dataset)
        return accuracy
        
###
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer
schedule_dict = {"inv":inv_lr_scheduler}

###
NLL_loss = torch.nn.NLLLoss().cuda() 
def Cross_Entropy(prob,lab):
    CE_loss = NLL_loss(torch.log(prob+1e-4), lab)
    return CE_loss

###Entropy Loss
def Entropy(prob):
    num_sam = prob.shape[0]
    Entropy = -(prob * (prob.log() + 1e-4)).sum(dim=1)
    return Entropy.mean()
