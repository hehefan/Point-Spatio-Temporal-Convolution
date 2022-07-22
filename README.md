# [PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences](https://openreview.net/pdf?id=O3bqkf_Puys)
![](https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/imgs/intro.png)

![](https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/imgs/equation.png)

## Introduction
Point cloud sequences are irregular and unordered in the spatial dimension while exhibiting regularities and order in the temporal dimension. Therefore, existing grid based convolutions for conventional video processing cannot be directly applied to spatio-temporal modeling of raw point cloud sequences. In the paper, we propose a point spatio-temporal (PST) convolution to achieve informative representations of point cloud sequences. The proposed PST convolution first disentangles space and time in point cloud sequences. Then, a spatial convolution is employed to capture the local structure of points in the 3D space, and a temporal convolution is used to model the dynamics of the spatial regions along the time dimension. 
![](https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/imgs/pstconv.png)
Furthermore, we incorporate the proposed PST convolution into a deep network, namely PSTNet, to extract features of 3D point cloud sequences in a spatio-temporally hierarchical manner. 
![](https://github.com/hehefan/Point-Spatio-Temporal-Convolution/blob/main/imgs/arch.png)

## Installation

The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 8.3.1, PyTorch v1.2, CUDA 10.2 and cuDNN v7.6.

Install PyTorch v1.2:
```
pip install torch==1.2.0 torchvision==0.4.0
```

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
cd modules
python setup.py install
```
To see if the compilation is successful, try to run `python modules/pst_convolutions.py` to see if a forward pass works.

Install [Mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html) for point cloud visualization (optional). Desktop is required.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{fan2021pstnet,
    title={PSTNet: Point Spatio-Temporal Convolution on Point Cloud Sequences},
    author={Hehe Fan and Xin Yu and Yuhang Ding and Yi Yang and Mohan Kankanhalli},
    booktitle={International Conference on Learning Representations},
    year={2021}
}
```

## Related Repos
1. PointNet++ PyTorch implementation: https://github.com/facebookresearch/votenet/tree/master/pointnet2
2. MeteorNet: https://github.com/xingyul/meteornet
3. 3DV: https://github.com/3huo/3DV-Action
4. P4Transformer: https://github.com/hehefan/P4Transformer
5. PointRNN (TensorFlow implementation): https://github.com/hehefan/PointRNN
6. PointRNN (PyTorch implementation): https://github.com/hehefan/PointRNN-PyTorch
7. Awesome Dynamic Point Cloud / Point Cloud Video / Point Cloud Sequence / 4D Point Cloud Analysis: https://github.com/hehefan/Awesome-Dynamic-Point-Cloud-Analysis
