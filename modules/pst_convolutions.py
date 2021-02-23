""" Point Point Spatio-Temporal (PST) Convolutions and Transposed Convolutions

From: "PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences"

Author: Hehe Fan
Date: July 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from typing import List

def kaiming_uniform(tensor, size):
    fan = size[1] * size[2] * size[3]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def uniform(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)

class PSTConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mid_planes: int,
                 out_planes: int,
                 spatial_kernel_size: [float, int],
                 temporal_kernel_size: int,
                 spatial_stride: int = 1,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 padding_mode: str = "zeros",
                 spatial_aggregation: str = "addition",
                 spatial_pooling: str = "max",
                 bias: bool = False,
                 batch_norm: bool = True):
        """
        Args:
            in_planes: C, number of point feature channels in the input. it is 0 if point features are not available.
            mid_planes: C_m, number of channels produced by the spatial convolution
            out_planes: C', number of channels produced by the temporal convolution
            spatial_kernel_size: (r, k), radius and nsamples
            temporal_kernel_size: odd
            spatial_stride: spatial sub-sampling rate, >= 1
            temporal_stride: controls the stride for the temporal cross correlation, >= 1
            temporal_padding:
            padding_mode: "zeros" or "replicate"
            spatial_aggregation: controls the way to aggregate point displacements and point features, "addition" or "multiplication"
            spatial_pooling: "max", "sum" or "avg"
            bias:
            batch_norm:
        """
        super().__init__()

        assert (padding_mode in ["zeros", "replicate"]), "PSTConv: 'padding_mode' should be 'zeros' or 'replicate'!"
        assert (spatial_aggregation in ["addition", "multiplication"]), "PSTConv: 'spatial_aggregation' should be 'addition' or 'multiplication'!"
        assert (spatial_pooling in ["max", "sum", "avg"]), "PSTConv: 'spatial_pooling' should be 'max', 'sum' or 'avg'!"

        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.out_planes = out_planes

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_radius = math.floor(temporal_kernel_size/2)
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.padding_mode = padding_mode

        self.spatial_aggregation = spatial_aggregation
        self.spatial_pooling = spatial_pooling

        if in_planes != 0:
            self.spatial_conv_f = nn.Conv2d(in_channels=in_planes, out_channels=mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
            kaiming_uniform(self.spatial_conv_f.weight, size=[mid_planes, in_planes+3, 1, 1])
            if bias:
                bound = 1 / math.sqrt(in_planes+3)
                uniform(self.spatial_conv_f.bias, -bound, bound)

        self.spatial_conv_d = nn.Conv2d(in_channels=3, out_channels=mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        kaiming_uniform(self.spatial_conv_d.weight, size=[mid_planes, in_planes+3, 1, 1])
        if bias:
            bound = 1 / math.sqrt(in_planes+3)
            uniform(self.spatial_conv_d.bias, -bound, bound)

        self.batch_norm = nn.BatchNorm1d(num_features=temporal_kernel_size*mid_planes) if batch_norm else False
        self.relu = nn.ReLU(inplace=True)

        self.temporal = nn.Conv1d(in_channels=temporal_kernel_size*mid_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, L, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, L, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)  # L
        npoints = xyzs.size(2)  # N

        if self.temporal_kernel_size > 1 and self.temporal_stride > 1:
            assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "PSTConv: Temporal parameter error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        if self.padding_mode == "zeros":
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]

            if self.in_planes != 0:
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
        else:   # "replicate"
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

            if self.in_planes != 0:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_radius, len(xyzs)-self.temporal_radius, self.temporal_stride):                                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            # spatial convolution
            spatial_features = []
            for i in range(t-self.temporal_radius, t+self.temporal_radius+1):
                neighbor_xyz = xyzs[i]

                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                           # (B, 3, N//spatial_stride, k)
                displacement = self.spatial_conv_d(displacement)                                                                    # (B, mid_planes, N//spatial_stride, k)

                if self.in_planes != 0:
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.spatial_conv_f(neighbor_feature_grouped)                                                         # (B, mid_planes, N//spatial_stride, k)

                    if self.spatial_aggregation == "addition":
                        spatial_feature = feature + displacement
                    else:
                        spatial_feature = feature * displacement

                else:
                    spatial_feature = displacement

                if self.spatial_pooling == 'max':
                    spatial_feature, _ = torch.max(input=spatial_feature, dim=-1, keepdim=False)                                    # (B, mid_planes, N//spatial_stride)
                elif self.spatial_pooling == 'sum':
                    spatial_feature = torch.sum(input=spatial_feature, dim=-1, keepdim=False)                                       # (B, mid_planes, N//spatial_stride)
                else:
                    spatial_feature = torch.mean(input=spatial_feature, dim=-1, keepdim=False)                                      # (B, mid_planes, N//spatial_stride)

                spatial_features.append(spatial_feature)

            spatial_features = torch.cat(tensors=spatial_features, dim=1, out=None)                                                 # (B, temporal_kernel_size*mid_planes, N//spatial_stride)

            # batch norm and relu
            if self.batch_norm:
                spatial_features = self.batch_norm(spatial_features)

            spatial_features = self.relu(spatial_features)

            # temporal convolution
            spatio_temporal_feature = self.temporal(spatial_features)

            new_xyzs.append(anchor_xyz)
            new_features.append(spatio_temporal_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

class PSTConvTranspose(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mid_planes: int,
                 out_planes: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 original_in_planes: int = 0,
                 bias: bool = False,
                 batch_norm: bool = True,
                 activation: bool = True):
        """
        Args:
            in_planes: C'. when point features are not available, in_planes is 0.
            mid_planes: C'_m
            out_planes: C"
            temporal_kernel_size: odd
            temporal_stride: controls the stride for the temporal cross correlation, >= 1
            temporal_padding: <=0, removes unnecessary temporal transposed features
            original_in_planes: C, used for skip connection from original points. when original point features are not available, original_in_planes is 0.
            bias: whether to use bias
            batch_norm: whether to use batch norm
            activation:
        """
        super().__init__()

        self.in_planes = in_planes
        self.mid_planes = mid_planes
        self.out_planes = out_planes

        # temporal parameters 
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_radius = math.floor(self.temporal_kernel_size/2)
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding

        # temporal transposed convolution
        self.temporal_conv = nn.Conv1d(in_channels=in_planes, out_channels=temporal_kernel_size*mid_planes, kernel_size=1, stride=1, padding=0, bias=bias)

        self.batch_norm = nn.BatchNorm1d(num_features=mid_planes) if batch_norm else False
        self.activation = nn.ReLU(inplace=True) if activation else False

        # spatial interpolation convolution
        self.spatial_conv = nn.Conv1d(in_channels=mid_planes+original_in_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


    def forward(self, xyzs: torch.Tensor, original_xyzs: torch.Tensor, features: torch.Tensor, original_features: torch.Tensor = None) -> torch.Tensor:
        r"""
        Parameters
        ----------
        xyzs : torch.Tensor
            (B, L', N', 3) tensor of the xyz positions of the convolved features
        original_xyzs : torch.Tensor
            (B, L,  N,  3) tensor of the xyz positions of the original points

        features : torch.Tensor
            (B, L', C', N') tensor of the features to be propigated to
        original_features : torch.Tensor
            (B, L,  C,  N) tensor of original point features for skip connection

        Returns
        -------
        new_features : torch.Tensor
            (B, L,  C", N) tensor of the features of the unknown features
        """

        L1 = original_xyzs.size(1)
        N1 = original_xyzs.size(2)

        L2 = xyzs.size(1)
        N2 = xyzs.size(2)

        if self.temporal_kernel_size > 1 and self.temporal_stride > 1:
            assert ((L2 - 1) * self.temporal_stride + sum(self.temporal_padding) + self.temporal_kernel_size == L1), "PSTConvTranspose: Temporal parameter error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
        features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

        new_xyzs = original_xyzs

        original_xyzs = torch.split(tensor=original_xyzs, split_size_or_sections=1, dim=1)
        original_xyzs = [torch.squeeze(input=original_xyz, dim=1).contiguous() for original_xyz in original_xyzs]

        if original_features is not None:
            original_features = torch.split(tensor=original_features, split_size_or_sections=1, dim=1)
            original_features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in original_features]

        # temporal transposed convolution
        temporal_trans_features = []
        for feature in features:
            feature = self.temporal_conv(feature)
            feature = torch.split(tensor=feature, split_size_or_sections=self.mid_planes, dim=1)
            temporal_trans_features.append(feature)

        # temporal interpolation
        temporal_interpolated_xyzs = []
        temporal_interpolated_features = []

        middles = []
        deltas = []
        for t2 in range(1, L2+1):
            middle = t2 + (t2-1)*(self.temporal_stride-1) + self.temporal_radius + self.temporal_padding[0]
            middles.append(middle)
            delta = range(middle - self.temporal_radius, middle + self.temporal_radius + self.temporal_padding[1] + 1)
            deltas.append(delta)

        for t1 in range(1, L1+1):
            seed_xyzs = []
            seed_features = []
            for t2 in range(L2):
                delta = deltas[t2]
                if t1 in delta:
                    seed_xyzs.append(xyzs[t2])
                    seed_feature = temporal_trans_features[t2][t1-middles[t2]+self.temporal_radius]
                    if self.batch_norm:
                        seed_feature = self.batch_norm(seed_feature)
                    if self.activation:
                        seed_feature = self.activation(seed_feature)
                    seed_features.append(seed_feature)
            seed_xyzs = torch.cat(seed_xyzs, dim=1)
            seed_features = torch.cat(seed_features, dim=2)
            temporal_interpolated_xyzs.append(seed_xyzs)
            temporal_interpolated_features.append(seed_features)

        # spatial interpolation
        new_features = []
        for t1 in range(L1):
            neighbor_xyz = temporal_interpolated_xyzs[t1]                                                               # [B, N', 3]
            anchor_xyz = original_xyzs[t1]                                                                              # [B, N,  3]

            dist, idx = pointnet2_utils.three_nn(anchor_xyz, neighbor_xyz)

            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(temporal_interpolated_features[t1], idx, weight)

            if original_features is not None:
                new_feature = torch.cat([interpolated_feats, original_features[t1]], dim=1)
            else:
                new_feature = interpolated_feats

            new_feature = self.spatial_conv(new_feature)

            new_features.append(new_feature)

        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features

if __name__ == '__main__':
    xyzs = torch.zeros(4, 8, 512, 3).cuda()
    features = torch.zeros(4, 8, 16, 512).cuda()

    conv = PSTConv(in_planes=16,
                   mid_planes=32,
                   out_planes=64,
                   spatial_kernel_size=[1.0, 3],
                   temporal_kernel_size=3,
                   spatial_stride=2,
                   temporal_stride=3,
                   temporal_padding=[1, 0],
                   padding_mode="replicate").cuda()

    new_xyzs, new_features = conv(xyzs, features)

    deconv =  PSTConvTranspose(in_planes=64,
                               mid_planes=128,
                               out_planes=256,
                               temporal_kernel_size=3,
                               temporal_stride=3,
                               temporal_padding=[-1, 0],
                               original_in_planes=16).cuda()

    out_xyzs, out_features = deconv(new_xyzs, xyzs, new_features, features)

    print("-----------------------------")
    print("Input:")
    print(xyzs.shape)
    print(features.shape)
    print("-----------------------------")
    print("PST convolution:")
    print(new_xyzs.shape)
    print(new_features.shape)
    print("-----------------------------")
    print("PST transposed convolution:")
    print(out_xyzs.shape)
    print(out_features.shape)
    print("-----------------------------")
