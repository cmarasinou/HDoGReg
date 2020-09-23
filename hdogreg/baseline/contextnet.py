# Implementation of neural network in
# Wang and Yang 2018

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable


def conv_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1), # paddding = 1 to preserve size
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    return pool


class globalNet(nn.Module):

    def __init__(self, in_dim, n_filt):
        super(globalNet,self).__init__()
        self.in_dim = in_dim
        self.n_filt = n_filt # = 32

        self.conv_1 = conv_block(self.in_dim,self.n_filt)
        self.pool_1 = maxpool()
        self.conv_2 = conv_block(self.n_filt, 2*self.n_filt)
        self.conv_3 = conv_block(2*self.n_filt,2*self.n_filt)
        self.pool_2 = maxpool()
        self.conv_4 = conv_block(2*self.n_filt,4*self.n_filt)
        self.conv_5 = conv_block(4*self.n_filt,4*self.n_filt)
        self.pool_3 = maxpool()
        self.conv_6 = conv_block(4*self.n_filt,8*self.n_filt)
        self.conv_7 = conv_block(8*self.n_filt,4*self.n_filt)
        self.pool_4 = maxpool()

    def forward(self,input):
        conv_1 = self.conv_1(input)
        pool_1 = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool_1)
        conv_3 = self.conv_3(conv_2)
        pool_2 = self.pool_2(conv_3)
        conv_4 = self.conv_4(pool_2)
        conv_5 = self.conv_5(conv_4)
        pool_3 = self.pool_3(conv_5)
        conv_6 = self.conv_6(pool_3)
        conv_7 = self.conv_7(conv_6)
        out = self.pool_4(conv_7)

        return out



class localNet(nn.Module):

    def __init__(self, in_dim, n_filt):
        super(localNet,self).__init__()
        self.in_dim = in_dim
        self.n_filt = n_filt # = 32

        self.conv_1 = conv_block(self.in_dim,self.n_filt)
        self.conv_2 = conv_block(self.n_filt, self.n_filt)
        self.pool = maxpool()
        self.conv_3 = conv_block(self.n_filt,2*self.n_filt)


    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        pool = self.pool(conv_2)
        out = self.conv_3(pool)

        return out


class contextNet(nn.Module):

    def __init__(self, in_dim, n_filt, dropout = 0.):
        super(contextNet,self).__init__()
        self.in_dim = in_dim
        self.n_filt = n_filt # = 32

        self.localnet = localNet(self.in_dim,self.n_filt)
        self.globalnet = globalNet(self.in_dim,self.n_filt)

        self.classifier = nn.Sequential(
            nn.Linear(4224, 128),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_large):
        input_small = input_large[:,:,43:43+9, 43:43+9]
        local_out = self.localnet(input_small)
        global_out = self.globalnet(input_large)

        # Merging networks
        n_batch = input_small.size(0)
        local_flat = local_out.view(n_batch,-1)
        global_flat = global_out.view(n_batch,-1)
        concat = torch.cat([local_flat,global_flat], dim=1)
        #concat = self.dropout(concat)
        # Feeding to classifier
        out = self.classifier(concat)

        return out