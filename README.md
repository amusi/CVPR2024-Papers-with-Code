# CVPR 2021 论文和开源项目合集(Papers with Code)

[CVPR 2021](http://cvpr2021.thecvf.com/) 论文和开源项目合集(papers with code)！

CVPR 2021 收录列表：http://cvpr2021.thecvf.com/sites/default/files/2021-03/accepted_paper_ids.txt

> 注1：欢迎各位大佬提交issue，分享CVPR 2021论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision

CVPR 2021 中奖群已成立！已经收录的同学，可以添加微信：**CVer9999**，请备注：**CVPR2021已收录+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群沟通开会等事宜。

## 【CVPR 2021 论文开源目录】

- [Backbone](#Backbone)
- [NAS](#NAS)
- [GAN](#GAN)
- [Visual Transformer](#Visual-Transformer)
- [Regularization](#Regularization)
- [无监督/自监督(Self-Supervised)](#Un/Self-Supervised)
- [半监督(Semi-Supervised)](#Semi-Supervised)
- [2D/遥感目标检测(Object Detection)](#Object-Detection)
- [单/多目标跟踪](#Object-Tracking)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#Medical-Image-Segmentation)
- [视频理解/行为识别(Video Understanding)](#Video-Understanding)
- [人脸识别(Face Recognition)](#Face-Recognition)
- [人脸检测(Face Detection)](#Face-Detection)
- [人脸活体检测(Face Anti-Spoofing)](#Face-Anti-Spoofing)
- [Deepfake检测(Deepfake Detection)](#Deepfake-Detection)
- [人脸年龄估计(Age-Estimation)](#Age-Estimation)
- [人体解析(Human Parsing)](#Human-Parsing)
- [2D/3D人体姿态估计(2D/3D Human Pose Estimation)](#Human-Pose-Estimation)
- [场景文本识别(Scene Text Recognition)](#Scene-Text-Recognition)
- [模型压缩/剪枝/量化](#Model-Compression)
- [超分辨率(Super-Resolution)](#Super-Resolution)
- [图像恢复(Image Restoration)](#Image-Restoration)
- [反光去除(Reflection Removal)](#Reflection-Removal)
- [3D目标检测(3D Object Detection)](#3D-Object-Detection)
- [3D语义分割(3D Semantic Segmentation)](#3D-Semantic-Segmentation)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D点云配准(3D Point Cloud Registration)](#3D-PointCloud-Registration)
- [3D点云补全(3D-Point-Cloud-Completion)](#3D-Point-Cloud-Completion)
- [6D位姿估计(6D Pose Estimation)](#6D-Pose-Estimation)
- [相机姿态估计(Camera Pose Estimation)](#Camera-Pose-Estimation)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [对抗样本(Adversarial-Examples)](#AE)
- [图像检索(Image Retrieval)](#Image-Retrieval)
- [Zero-Shot Learning](#Zero-Shot-Learning)
- [联邦学习(Federated Learning)](#Federated-Learning)
- [视频插帧(Video Frame Interpolation)](#Video-Frame-Interpolation)
- [视觉推理(Visual Reasoning)](#Visual-Reasoning)
- [视图合成(Visual Synthesis)](#Visual-Synthesis)
- [Domain Generalization](#Domain-Generalization)
- ["人-物"交互(HOI)检测](#HOI)
- [阴影去除(Shadow Removal)](#Shadow-Removal)
- [虚拟试衣](#Virtual-Try-On)
- [数据集(Datasets)](#Datasets)
- [其他(Others)](#Others)
- [待添加(TODO)](#TO-DO)
- [不确定中没中(Not Sure)](#Not-Sure)

<a name="Backbone"></a>

# Backbone

**ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network**

- Paper: https://arxiv.org/abs/2007.00992
- Code:  https://github.com/clovaai/rexnet

**Involution: Inverting the Inherence of Convolution for Visual Recognition**

- Paper: https://github.com/d-li14/involution
- Code: https://arxiv.org/abs/2103.06255

**Coordinate Attention for Efficient Mobile Network Design**

- Paper:  https://arxiv.org/abs/2103.02907
- Code: https://github.com/Andrew-Qibin/CoordAttention

**Inception Convolution with Efficient Dilation Search**

- Paper:  https://arxiv.org/abs/2012.13587 
- Code: https://github.com/yifan123/IC-Conv

**RepVGG: Making VGG-style ConvNets Great Again**

- Paper: https://arxiv.org/abs/2101.03697
- Code: https://github.com/DingXiaoH/RepVGG

<a name="NAS"></a>

# NAS

**Searching by Generating: Flexible and Efficient One-Shot NAS with Architecture Generator**

- Paper: https://arxiv.org/abs/2103.07289
- Code: https://github.com/eric8607242/SGNAS

**OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection**

- Paper: https://arxiv.org/abs/2103.04507
- Code: https://github.com/VDIGPKU/OPANAS

**Inception Convolution with Efficient Dilation Search**

- Paper:  https://arxiv.org/abs/2012.13587 
- Code: None

<a name="GAN"></a>

# GAN

**HumanGAN: A Generative Model of Humans Images**

- Paper: https://arxiv.org/abs/2103.06902
- Code: None

**ID-Unet: Iterative Soft and Hard Deformation for View Synthesis**

- Paper: https://arxiv.org/abs/2103.02264
- Code: https://github.com/MingyuY/Iterative-view-synthesis

**CoMoGAN: continuous model-guided image-to-image translation**

- Paper: https://arxiv.org/abs/2103.06879
- Code: https://github.com/cv-rits/CoMoGAN

**Training Generative Adversarial Networks in One Stage**

- Paper: https://arxiv.org/abs/2103.00430
- Code: None

**Closed-Form Factorization of Latent Semantics in GANs**

- Homepage: https://genforce.github.io/sefa/
- Paper: https://arxiv.org/abs/2007.06600
- Code: https://github.com/genforce/sefa

**Anycost GANs for Interactive Image Synthesis and Editing**

- Paper: https://arxiv.org/abs/2103.03243
- Code: https://github.com/mit-han-lab/anycost-gan

**Image-to-image Translation via Hierarchical Style Disentanglement**

- Paper: https://arxiv.org/abs/2103.01456
- Code: https://github.com/imlixinyang/HiSD

<a name="Visual Transformer"></a>

# Visual Transformer

**End-to-End Video Instance Segmentation with Transformers**

- Paper(Oral): https://arxiv.org/abs/2011.14503
- Code: https://github.com/Epiphqny/VisTR

**UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**

- Paper(Oral): https://arxiv.org/abs/2011.09094
- Code: https://github.com/dddzg/up-detr

**End-to-End Human Object Interaction Detection with HOI Transformer**

- Paper: https://arxiv.org/abs/2103.04503
- Code: https://github.com/bbepoch/HoiTransformer

**Transformer Interpretability Beyond Attention Visualization** 

- Paper: https://arxiv.org/abs/2012.09838
- Code: https://github.com/hila-chefer/Transformer-Explainability 

<a name="Regularization"></a>

# Regularization

**Regularizing Neural Networks via Adversarial Model Perturbation**

- Paper: https://arxiv.org/abs/2010.04925
- Code: https://github.com/hiyouga/AMP-Regularizer

<a name="Un/Self-Supervised"></a>

# 无监督/自监督(Un/Self-Supervised)

**Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning**

- Homepage: https://fingerrec.github.io/index_files/jinpeng/papers/CVPR2021/project_website.html
- Paper: https://arxiv.org/abs/2009.05769
- Code: https://github.com/FingerRec/BE

**Spatially Consistent Representation Learning**

- Paper: https://arxiv.org/abs/2103.06122
- Code: None

**VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples**

- Paper: https://arxiv.org/abs/2103.05905
- Code: https://github.com/tinapan-pt/VideoMoCo

**Exploring Simple Siamese Representation Learning**

- Paper(Oral): https://arxiv.org/abs/2011.10566
- Code: None

**Dense Contrastive Learning for Self-Supervised Visual Pre-Training**

- Paper(Oral): https://arxiv.org/abs/2011.09157
- Code: https://github.com/WXinlong/DenseCL

<a name="Semi-Supervised"></a>

# 半监督学习(Semi-Supervised )

**Adaptive Consistency Regularization for Semi-Supervised Transfer Learning**

- Paper: https://arxiv.org/abs/2103.02193
- Code: https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning

<a name="Object-Detection"></a>

# 2D/遥感目标检测(Object Detection)

## 2D目标检测

**OPANAS: One-Shot Path Aggregation Network Architecture Search for Object Detection**

- Paper: https://arxiv.org/abs/2103.04507
- Code: https://github.com/VDIGPKU/OPANAS

**YOLOF：You Only Look One-level Feature**

- Paper: https://arxiv.org/abs/2103.09460
- Code: https://github.com/megvii-model/YOLOF

**UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**

- Paper(Oral): https://arxiv.org/abs/2011.09094
- Code: https://github.com/dddzg/up-detr

**General Instance Distillation for Object Detection**

- Paper: https://arxiv.org/abs/2103.02340
- Code: None

**Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection**

- Paper: https://arxiv.org/abs/2103.01903
- Code: None

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**

- Homepage: http://rl.uni-freiburg.de/research/multimodal-distill
- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

**Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection**

- Paper: https://arxiv.org/abs/2011.12885
- Code: https://github.com/implus/GFocalV2

**Multiple Instance Active Learning for Object Detection**

- Paper: https://github.com/yuantn/MIAL/raw/master/paper.pdf
- Code: https://github.com/yuantn/MIAL

**Towards Open World Object Detection**

- Paper: https://arxiv.org/abs/2103.02603
- Code: https://github.com/JosephKJ/OWOD

## Few-Shot目标检测

**Few-Shot Object Detection via Contrastive Proposal Encoding**

- Paper: https://arxiv.org/abs/2103.05950
- Code: https://github.com/MegviiDetection/FSCE 

## 旋转目标检测

**ReDet: A Rotation-equivariant Detector for Aerial Object Detection**

- Paper: https://arxiv.org/abs/2103.07733

- Code: https://github.com/csuhan/ReDet

<a name="Object-Tracking"></a>

# 单/多目标跟踪(Object Tracking)

**Track to Detect and Segment: An Online Multi-Object Tracker**

- Homepage: https://jialianwu.com/projects/TraDeS.html
- Paper: https://arxiv.org/abs/2103.08808
- Code: https://github.com/JialianW/TraDeS

<a name="Instance-Segmentation"></a>

# 实例分割(Instance Segmentation)

**End-to-End Video Instance Segmentation with Transformers**

- Paper(Oral): https://arxiv.org/abs/2011.14503
- Code: https://github.com/Epiphqny/VisTR

**Zero-shot instance segmentation（Not Sure）**

- Paper: None
- Code: https://github.com/CVPR2021-pape-id-1395/CVPR2021-paper-id-1395

<a name="Panoptic-Segmentation"></a>

# 全景分割(Panoptic Segmentation)

**Fully Convolutional Networks for Panoptic Segmentation**

- Paper: https://arxiv.org/abs/2012.00720

- Code: https://github.com/yanwei-li/PanopticFCN

**Cross-View Regularization for Domain Adaptive Panoptic Segmentation**

- Paper: https://arxiv.org/abs/2103.02584
- Code: None

<a name="Medical-Image-Segmentation"></a>

# 医学图像分割

**FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space**

- Paper: https://arxiv.org/abs/2103.06030
- Code: https://github.com/liuquande/FedDG-ELCFS

<a name="Video-Understanding"></a>

# 视频理解/行为识别(Video Understanding)

**ACTION-Net: Multipath Excitation for Action Recognition**

- Paper: https://arxiv.org/abs/2103.07372
- Code: https://github.com/V-Sense/ACTION-Net

**Removing the Background by Adding the Background: Towards Background Robust Self-supervised Video Representation Learning**

- Homepage: https://fingerrec.github.io/index_files/jinpeng/papers/CVPR2021/project_website.html
- Paper: https://arxiv.org/abs/2009.05769
- Code: https://github.com/FingerRec/BE

**TDN: Temporal Difference Networks for Efficient Action Recognition**

- Paper: https://arxiv.org/abs/2012.10071
- Code: https://github.com/MCG-NJU/TDN

<a name="Face-Recognition"></a>

# 人脸识别(Face Recognition)

**MagFace: A Universal Representation for Face Recognition and Quality Assessment**

- Paper(Oral): https://arxiv.org/abs/2103.06627
- Code: https://github.com/IrvingMeng/MagFace

**WebFace260M: A Benchmark Unveiling the Power of Million-Scale Deep Face Recognition**

- Homepage: https://www.face-benchmark.org/
- Paper: https://arxiv.org/abs/2103.04098 
- Dataset: https://www.face-benchmark.org/

**When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework**

- Paper(Oral): https://arxiv.org/abs/2103.01520
- Code: https://github.com/Hzzone/MTLFace
- Dataset: https://github.com/Hzzone/MTLFace

<a name="Face-Detection"></a>

# 人脸检测(Face Detection)

**CRFace: Confidence Ranker for Model-Agnostic Face Detection Refinement**

- Paper: https://arxiv.org/abs/2103.07017
- Code: None

<a name="Face-Anti-Spoofing"></a>

# 人脸活体检测(Face Anti-Spoofing)

**Cross Modal Focal Loss for RGBD Face Anti-Spoofing**

- Paper: https://arxiv.org/abs/2103.00948
- Code: None

<a name="Deepfake-Detection"></a>

# Deepfake检测(Deepfake Detection)

**Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain**

- Paper：https://arxiv.org/abs/2103.01856
- Code: None

**Multi-attentional Deepfake Detection**

- Paper：https://arxiv.org/abs/2103.02406
- Code: None

<a name="Age-Estimation"></a>

# 人脸年龄估计(Age Estimation)

**PML: Progressive Margin Loss for Long-tailed Age Classification**

- Paper: https://arxiv.org/abs/2103.02140
- Code: None

<a name="Human-Parsing"></a>

# 人体解析(Human Parsing)

**Differentiable Multi-Granularity Human Representation Learning for Instance-Aware Human Semantic Parsing**

- Paper: https://arxiv.org/abs/2103.04570
- Code: https://github.com/tfzhou/MG-HumanParsing

<a name="Human-Pose-Estimation"></a>

# 2D/3D人体姿态估计(2D/3D Human Pose Estimation)

## 2D 人体姿态估计

**DCPose: Deep Dual Consecutive Network for Human Pose Estimation**

-  Paper: https://arxiv.org/abs/2103.07254
- Code: https://github.com/Pose-Group/DCPose 

## 3D 人体姿态估计

**HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation**

- Homepage: https://jeffli.site/HybrIK/ 
- Paper: https://arxiv.org/abs/2011.14672
- Code: https://github.com/Jeff-sjtu/HybrIK

<a name="Scene-Text-Recognition"></a>

# 场景文本识别(Scene Text Recognition)

**Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition**

- Paper: https://arxiv.org/abs/2103.06495
- Code: https://github.com/FangShancheng/ABINet

<a name="Model-Compression"></a>

# 模型压缩/剪枝/量化

## 模型量化

**Learnable Companding Quantization for Accurate Low-bit Neural Networks**

- Paper: https://arxiv.org/abs/2103.07156
- Code: None

<a name="Super-Resolution"></a>

# 超分辨率(Super-Resolution)

**ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic**

- Paper: https://arxiv.org/abs/2103.04039
- Code: https://github.com/Xiangtaokong/ClassSR

**AdderSR: Towards Energy Efficient Image Super-Resolution**

- Paper: https://arxiv.org/abs/2009.08891
- Code: None

<a name="Image-Restoration"></a>

# 图像恢复(Image Restoration)

**Multi-Stage Progressive Image Restoration**

- Paper: https://arxiv.org/abs/2102.02808
- Code: https://github.com/swz30/MPRNet

<a name="Reflection-Removal"></a>

# 反光去除(Reflection Removal)

**Robust Reflection Removal with Reflection-free Flash-only Cues**

- Paper: https://arxiv.org/abs/2103.04273
- Code: https://github.com/ChenyangLEI/flash-reflection-removal

<a name="3D-Object-Detection"></a>

# 3D目标检测(3D Object Detection)

**SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud**

- Paper: None
- Code: https://github.com/Vegeta2020/SE-SSD

**Center-based 3D Object Detection and Tracking**

- Paper: https://arxiv.org/abs/2006.11275
- Code: https://github.com/tianweiy/CenterPoint

**Categorical Depth Distribution Network for Monocular 3D Object Detection**

- Paper: https://arxiv.org/abs/2103.01100
- Code: None

<a name="3D-Semantic-Segmentation"></a>

# 3D语义分割(3D Semantic Segmentation)

**Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion**

- Paper: https://arxiv.org/abs/2103.07074
- Code: https://github.com/ShiQiu0419/BAAF-Net

**Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation**

- Paper: https://arxiv.org/abs/2011.10033
- Code:  https://github.com/xinge008/Cylinder3D 

 **Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges**

- Homepage: https://github.com/QingyongHu/SensatUrban
- Paper: http://arxiv.org/abs/2009.03137
- Code: https://github.com/QingyongHu/SensatUrban
- Dataset: https://github.com/QingyongHu/SensatUrban

<a name="3D-Object-Tracking"></a>

# 3D目标跟踪(3D Object Trancking)

**Center-based 3D Object Detection and Tracking**

- Paper: https://arxiv.org/abs/2006.11275
- Code: https://github.com/tianweiy/CenterPoint

<a name="3D-PointCloud-Registration"></a>

# 3D点云配准(3D Point Cloud Registration)

**PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency**

- Paper: https://arxiv.org/abs/2103.05465
- Code: https://github.com/XuyangBai/PointDSC 

**PREDATOR: Registration of 3D Point Clouds with Low Overlap**

- Paper: https://arxiv.org/abs/2011.13005
- Code: https://github.com/ShengyuH/OverlapPredator

<a name="3D-Point-Cloud-Completion"></a>

# 3D点云补全(3D Point Cloud Completion)

**Style-based Point Generator with Adversarial Rendering for Point Cloud Completion**

- Paper: https://arxiv.org/abs/2103.02535
- Code: None

<a name="6D-Pose-Estimation"></a>

# 6D位姿估计(6D Pose Estimation)

**GDR-Net: Geometry-Guided Direct Regression Network for Monocular 6D Object Pose Estimation**

- Paper: http://arxiv.org/abs/2102.12145
- code: https://git.io/GDR-Net

**FFB6D: A Full Flow Bidirectional Fusion Network for 6D Pose Estimation**

- Paper: https://arxiv.org/abs/2103.02242
- Code: https://github.com/ethnhe/FFB6D

<a name="Camera-Pose-Estimation"></a>

# 相机姿态估计

**Back to the Feature: Learning Robust Camera Localization from Pixels to Pose**

- Paper: https://arxiv.org/abs/2103.09213
- Code: https://github.com/cvg/pixloc

<a name="Depth-Estimation"></a>

# 深度估计

**Beyond Image to Depth: Improving Depth Prediction using Echoes**

- Homepage: https://krantiparida.github.io/projects/bimgdepth.html
- Paper: https://arxiv.org/abs/2103.08468
- Code: https://github.com/krantiparida/beyond-image-to-depth

**S3: Learnable Sparse Signal Superdensity for Guided Depth Estimation**

- Paper: https://arxiv.org/abs/2103.02396
- Code: None

**Depth from Camera Motion and Object Detection**

- Paper: https://arxiv.org/abs/2103.01468
- Code: https://github.com/griffbr/ODMD
- Dataset: https://github.com/griffbr/ODMD

<a name="AE"></a>

# 对抗样本

**Natural Adversarial Examples**

- Paper: https://arxiv.org/abs/1907.07174
- Code: https://github.com/hendrycks/natural-adv-examples

<a name="Image-Retrieval"></a>

# 图像检索(Image Retrieval)

**QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval**

- Paper: https://arxiv.org/abs/2103.02927
- Code: None

<a name="Zero-Shot-Learning"></a>

#  Zero-Shot Learning

**Counterfactual Zero-Shot and Open-Set Visual Recognition**

- Paper: https://arxiv.org/abs/2103.00887
- Code: https://github.com/yue-zhongqi/gcm-cf

<a name="Federated-Learning"></a>

# 联邦学习(Federated Learning)

**FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space**

- Paper: https://arxiv.org/abs/2103.06030
- Code: https://github.com/liuquande/FedDG-ELCFS

<a name="Video-Frame-Interpolation"></a>

# 视频插帧(Video Frame Interpolation)

**FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation**

- Homepage: https://tarun005.github.io/FLAVR/

- Paper: https://arxiv.org/abs/2012.08512
- Code: https://github.com/tarun005/FLAVR

<a name="Visual-Reasoning"></a>

# 视觉推理(Visual Reasoning)

**Transformation Driven Visual Reasoning**

- homepage: https://hongxin2019.github.io/TVR/
- Paper: https://arxiv.org/abs/2011.13160
- Code: https://github.com/hughplay/TVR

<a name="Visual-Synthesis"></a>

# 视图合成(View Synthesis)

**NeX: Real-time View Synthesis with Neural Basis Expansion**

- Homepage: https://nex-mpi.github.io/
- Paper(Oral): https://arxiv.org/abs/2103.05606

<a name="Domain-Generalization"></a>

# DomainGeneralization

**FSDR: Frequency Space Domain Randomization for Domain Generalization**

- Paper: https://arxiv.org/abs/2103.02370
- Code: None

<a name="HOI"></a>

# "人-物"交互(HOI)检测

**Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information**

- Paper: https://arxiv.org/abs/2103.05399
- Code: https://github.com/hitachi-rd-cv/qpic

**Reformulating HOI Detection as Adaptive Set Prediction**

- Paper: https://arxiv.org/abs/2103.05983
- Code: https://github.com/yoyomimi/AS-Net

**Detecting Human-Object Interaction via Fabricated Compositional Learning**

- Paper: https://arxiv.org/abs/2103.08214
- Code: https://github.com/zhihou7/FCL

**End-to-End Human Object Interaction Detection with HOI Transformer**

- Paper: https://arxiv.org/abs/2103.04503
- Code: https://github.com/bbepoch/HoiTransformer

<a name="Shadow-Removal"></a>

# 阴影去除(Shadow Removal)

**Auto-Exposure Fusion for Single-Image Shadow Removal**

- Paper: https://arxiv.org/abs/2103.01255
- Code: https://github.com/tsingqguo/exposure-fusion-shadow-removal

<a name="Virtual-Try-On"></a>

# 虚拟换衣(Virtual Try-On)

**Parser-Free Virtual Try-on via Distilling Appearance Flows**

**基于外观流蒸馏的无需人体解析的虚拟换装**

- Paper: https://arxiv.org/abs/2103.04559
- Code: https://github.com/geyuying/PF-AFN 

<a name="Datasets"></a>

# 数据集(Datasets)

**Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food**

- Paper: https://arxiv.org/abs/2103.03375
- Dataset: None

 **Towards Semantic Segmentation of Urban-Scale 3D Point Clouds: A Dataset, Benchmarks and Challenges**

- Homepage: https://github.com/QingyongHu/SensatUrban
- Paper: http://arxiv.org/abs/2009.03137
- Code: https://github.com/QingyongHu/SensatUrban
- Dataset: https://github.com/QingyongHu/SensatUrban

**When Age-Invariant Face Recognition Meets Face Age Synthesis: A Multi-Task Learning Framework**

- Paper(Oral): https://arxiv.org/abs/2103.01520
- Code: https://github.com/Hzzone/MTLFace
- Dataset: https://github.com/Hzzone/MTLFace

**Depth from Camera Motion and Object Detection**

- Paper: https://arxiv.org/abs/2103.01468
- Code: https://github.com/griffbr/ODMD
- Dataset: https://github.com/griffbr/ODMD

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**

- Homepage: http://rl.uni-freiburg.de/research/multimodal-distill
- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

**Scan2Cap: Context-aware Dense Captioning in RGB-D Scans**

- Paper: https://arxiv.org/abs/2012.02206
- Code: https://github.com/daveredrum/Scan2Cap

- Dataset: https://github.com/daveredrum/ScanRefer

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**

- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill
- Dataset: http://rl.uni-freiburg.de/research/multimodal-distill

<a name="Others"></a>

# 其他(Others)

**Knowledge Evolution in Neural Networks**

- Paper(Oral): https://arxiv.org/abs/2103.05152
- Code: https://github.com/ahmdtaha/knowledge_evolution

**Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning**

- Paper: https://arxiv.org/abs/2103.02148
- Code: https://github.com/guopengf/FLMRCM

**SGP: Self-supervised Geometric Perception**

- Oral

- Paper: https://arxiv.org/abs/2103.03114
- Code: https://github.com/theNded/SGP

**Multi-institutional Collaborations for Improving Deep Learning-based Magnetic Resonance Image Reconstruction Using Federated Learning**

- Paper: https://arxiv.org/abs/2103.02148
- Code: https://github.com/guopengf/FLMRCM

**Diffusion Probabilistic Models for 3D Point Cloud Generation**

- Paper: https://arxiv.org/abs/2103.01458
- Code: https://github.com/luost26/diffusion-point-cloud

**Scan2Cap: Context-aware Dense Captioning in RGB-D Scans**

- Paper: https://arxiv.org/abs/2012.02206
- Code: https://github.com/daveredrum/Scan2Cap

- Dataset: https://github.com/daveredrum/ScanRefer

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**

- Paper: https://arxiv.org/abs/2103.01353
- Code: http://rl.uni-freiburg.de/research/multimodal-distill

- Dataset: http://rl.uni-freiburg.de/research/multimodal-distill

<a name="TO-DO"></a>

# 待添加(TODO)

- [重磅！腾讯优图20篇论文入选CVPR 2021](https://mp.weixin.qq.com/s/McAtOVh0osWZ3uppEoHC8A)
- [MePro团队三篇论文被CVPR 2021接收](https://mp.weixin.qq.com/s/GD5Zb6u_MQ8GZIAGeCGo3Q)

<a name="Not-Sure"></a>

# 不确定中没中(Not Sure)

**CT Film Recovery via Disentangling Geometric Deformation and Photometric Degradation: Simulated Datasets and Deep Models**

- Paper: none
- Code: https://github.com/transcendentsky/Film-Recovery

**Toward Explainable Reflection Removal with Distilling and Model Uncertainty**

- Paper: none
- Code: https://github.com/ytpeng-aimlab/CVPR-2021-Toward-Explainable-Reflection-Removal-with-Distilling-and-Model-Uncertainty

**DeepOIS: Gyroscope-Guided Deep Optical Image Stabilizer Compensation**

- Paper: none
- Code: https://github.com/lhaippp/DeepOIS

**Exploring Adversarial Fake Images on Face Manifold**

- Paper: none
- Code: https://github.com/ldz666666/Style-atk

**Uncertainty-Aware Semi-Supervised Crowd Counting via Consistency-Regularized Surrogate Task**

- Paper: none
- Code: https://github.com/yandamengdanai/Uncertainty-Aware-Semi-Supervised-Crowd-Counting-via-Consistency-Regularized-Surrogate-Task

**Temporal Contrastive Graph for Self-supervised Video Representation Learning**

- Paper: none
- Code: https://github.com/YangLiu9208/TCG

**Boosting Monocular Depth Estimation Models to High-Resolution via Context-Aware Patching**

- Paper: none
- Code: https://github.com/ouranonymouscvpr/cvpr2021_ouranonymouscvpr

**Fast and Memory-Efficient Compact Bilinear Pooling**

- Paper: none
- Code: https://github.com/cvpr2021kp2/cvpr2021kp2

**Identification of Empty Shelves in Supermarkets using Domain-inspired Features with Structural Support Vector Machine**

- Paper: none
- Code: https://github.com/gapDetection/cvpr2021

 **Estimating A Child's Growth Potential From Cephalometric X-Ray Image via Morphology-Aware Interactive Keypoint Estimation** 

- Paper: none
- Code: https://github.com/interactivekeypoint2020/Morph

https://github.com/ShaoQiangShen/CVPR2021

https://github.com/gillesflash/CVPR2021

https://github.com/anonymous-submission1991/BaLeNAS

https://github.com/cvpr2021dcb/cvpr2021dcb

https://github.com/anonymousauthorCV/CVPR2021_PaperID_8578

https://github.com/AldrichZeng/FreqPrune

https://github.com/Anonymous-AdvCAM/Anonymous-AdvCAM

https://github.com/ddfss/datadrive-fss

