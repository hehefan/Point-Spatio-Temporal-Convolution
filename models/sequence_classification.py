import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from pst_convolutions import PSTConv

class MSRAction(nn.Module):
    def __init__(self, radius=1.5, nsamples=3*3, num_classes=20):
        super(MSRAction, self).__init__()

        self.conv1 =  PSTConv(in_planes=0,
                              mid_planes=45,
                              out_planes=64,
                              spatial_kernel_size=[radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2a = PSTConv(in_planes=64,
                              mid_planes=96,
                              out_planes=128,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv2b = PSTConv(in_planes=128,
                              mid_planes=192,
                              out_planes=256,
                              spatial_kernel_size=[2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3a = PSTConv(in_planes=256,
                              mid_planes=384,   # mistake 284
                              out_planes=512,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=2,
                              temporal_stride=2,
                              temporal_padding=[1,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv3b = PSTConv(in_planes=512,
                              mid_planes=768,
                              out_planes=1024,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=3,
                              spatial_stride=1,
                              temporal_stride=1,
                              temporal_padding=[1,1],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.conv4 =  PSTConv(in_planes=1024,
                              mid_planes=1536,
                              out_planes=2048,
                              spatial_kernel_size=[2*2*radius, nsamples],
                              temporal_kernel_size=1,
                              spatial_stride=2,
                              temporal_stride=1,
                              temporal_padding=[0,0],
                              spatial_aggregation="multiplication",
                              spatial_pooling="sum")

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, xyzs):

        new_xys, new_features = self.conv1(xyzs, None)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv2b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3a(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv3b(new_xys, new_features)
        new_features = F.relu(new_features)

        new_xys, new_features = self.conv4(new_xys, new_features)               # (B, L, C, N)

        new_features = torch.mean(input=new_features, dim=-1, keepdim=False)    # (B, L, C)

        new_feature = torch.max(input=new_features, dim=1, keepdim=False)[0]    # (B, C)

        out = self.fc(new_feature)

        return out
