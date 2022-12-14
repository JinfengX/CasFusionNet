# CasFusionNet: A Cascaded Network for Point Cloud Semantic Scene Completion by Dense Feature Fusion
by Jinfeng Xu, [Xianzhi Li](https://nini-lxz.github.io/), Yuan Tang, Qiao Yu, Yixue Hao, Long Hu, Min Chen

<div style="text-align: center;">
    <img style="border-radius: 0.3125em;
    width: 98%;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=./figures/teaser.png alt="">
    <br>
</div>


## Introduction
This repository is for our Advancement of Artificial Intelligence (AAAI) 2023 paper 
'CasFusionNet: A Cascaded Network for Point Cloud Semantic Scene Completion by Dense Feature Fusion'. 
In this paper, we present a novel cascaded network for point cloud semantic scene completion (PC-SSC), 
which aims to infer both semantics and geometry from a partial 3D scene. 
Opposed to voxel-based methods, our network only consume point cloud. 
We designed three module to perform PC-SSC, i.e., 
(i) a global completion module (GCM) to produce an upsampled and completed but coarse point set, 
(ii) a semantic segmentation module (SSM) to predict the per-point semantic labels of the completed points generated by GCM, 
and (iii) a local refinement module (LRM) to further refine the coarse completed points and the associated labels in a local perspective. 
To fully exploit the connection between scene completion and semantic segmentation task, 
we associate above three modules via dense feature fusion in each level, and cascade a total of four levels, 
where we also employ skip connection and feature fusion between each level for sufficient information usage. 
We evaluate proposed method on our compiled two point-based datasets and 
compared to state-of-the-art methods in terms of both scene completion and semantic segmentation.


## TODO
- [ ] Environment build
- [ ] Data preparation and datasets release
- [ ] Code release
- [ ] Model parameter release
- [ ] Other tools release


## Coming soon


## Questions
Please contact <a href="mailto:jinfengxu.edu@gmail.com">jinfengxu.edu@gmail.com</a>