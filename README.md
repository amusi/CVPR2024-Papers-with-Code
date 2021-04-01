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
- [VAE](#VAE)
- [Visual Transformer](#Visual-Transformer)
- [Regularization](#Regularization)
- [SLAM](#SLAM)
- [长尾分布(Long-Tailed)](#Long-Tailed)
- [数据增广(Data Augmentation)](#DA)
- [无监督/自监督(Self-Supervised)](#Un/Self-Supervised)
- [半监督(Semi-Supervised)](#Semi-Supervised)
- [胶囊网络(Capsule Network)](#Capsule-Network)
- [2D目标检测(Object Detection)](#Object-Detection)
- [单/多目标跟踪(Object Tracking)](#Object-Tracking)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#Medical-Image-Segmentation)
- [交互式视频目标分割(Interactive-Video-Object-Segmentation)](#IVOS)
- [显著性检测(Saliency Detection)](#Saliency-Detection)
- [行人搜索(Person Search)](#Person-Search)
- [视频理解/行为识别(Video Understanding)](#Video-Understanding)
- [人脸识别(Face Recognition)](#Face-Recognition)
- [人脸检测(Face Detection)](#Face-Detection)
- [人脸活体检测(Face Anti-Spoofing)](#Face-Anti-Spoofing)
- [Deepfake检测(Deepfake Detection)](#Deepfake-Detection)
- [人脸年龄估计(Age-Estimation)](#Age-Estimation)
- [人脸表情识别(Facial-Expression-Recognition)](#FER)
- [Deepfakes](#Deepfakes)
- [人体解析(Human Parsing)](#Human-Parsing)
- [2D/3D人体姿态估计(2D/3D Human Pose Estimation)](#Human-Pose-Estimation)
- [动物姿态估计(Animal Pose Estimation)](#Animal-Pose-Estimation)
- [Human Volumetric Capture](#Human-Volumetric-Capture)
- [场景文本识别(Scene Text Recognition)](#Scene-Text-Recognition)
- [图像压缩(Image Compression)](#Image-Compression)
- [模型压缩/剪枝/量化](#Model-Compression)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [超分辨率(Super-Resolution)](#Super-Resolution)
- [图像恢复(Image Restoration)](#Image-Restoration)
- [图像补全(Image Inpainting)](#Image-Inpainting)
- [图像编辑(Image Editing)](#Image-Editing)
- [图像匹配(Image Matching)](#Image-Matching)
- [反光去除(Reflection Removal)](#Reflection-Removal)
- [3D点云分类(3D Point Clouds Classification)](#3D-C)
- [3D目标检测(3D Object Detection)](#3D-Object-Detection)
- [3D语义分割(3D Semantic Segmentation)](#3D-Semantic-Segmentation)
- [3D全景分割(3D Panoptic Segmentation)](#3D-Panoptic-Segmentation)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D点云配准(3D Point Cloud Registration)](#3D-PointCloud-Registration)
- [3D点云补全(3D-Point-Cloud-Completion)](#3D-Point-Cloud-Completion)
- [6D位姿估计(6D Pose Estimation)](#6D-Pose-Estimation)
- [相机姿态估计(Camera Pose Estimation)](#Camera-Pose-Estimation)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [对抗样本(Adversarial-Examples)](#AE)
- [图像检索(Image Retrieval)](#Image-Retrieval)
- [视频检索(Video Retrieval)](#Video-Retrieval)
- [跨模态检索(Cross-modal Retrieval)](#Cross-modal-Retrieval) 
- [Zero-Shot Learning](#Zero-Shot-Learning)
- [联邦学习(Federated Learning)](#Federated-Learning)
- [视频插帧(Video Frame Interpolation)](#Video-Frame-Interpolation)
- [视觉推理(Visual Reasoning)](#Visual-Reasoning)
- [视图合成(Visual Synthesis)](#Visual-Synthesis)
- [Domain Generalization](#Domain-Generalization)
- [Open-Set Recognition](#Open-Set-Recognition)
- [Adversarial Attack](#Adversarial-Attack)
- ["人-物"交互(HOI)检测](#HOI)
- [阴影去除(Shadow Removal)](#Shadow-Removal)
- [虚拟试衣](#Virtual-Try-On)
- [数据集(Datasets)](#Datasets)
- [其他(Others)](#Others)
- [待添加(TODO)](#TO-DO)
- [不确定中没中(Not Sure)](#Not-Sure)

<a name="Backbone"></a>

# Backbone

**Diverse Branch Block: Building a Convolution as an Inception-like Unit**

- Paper: https://arxiv.org/abs/2103.13425

- Code: https://github.com/DingXiaoH/DiverseBranchBlock

**Scaling Local Self-Attention For Parameter Efficient Visual Backbones**

- Paper(Oral): https://arxiv.org/abs/2103.12731

- Code: None

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

**DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation**

- Paper(Oral): https://arxiv.org/abs/2103.15954
- Code: None

**HR-NAS: Searching Efficient High-Resolution Neural Architectures with Transformers**

- Paper(Oral): None
- Code: https://github.com/dingmyu/HR-NAS

**Neural Architecture Search with Random Labels**

- Paper: https://arxiv.org/abs/2101.11834
- Code: None

**Towards Improving the Consistency, Efficiency, and Flexibility of Differentiable Neural Architecture Search**

- Paper: https://arxiv.org/abs/2101.11342
- Code: None

**Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation**

- Paper: None
- Code: None

**Prioritized Architecture Sampling with Monto-Carlo Tree Search**

- Paper: https://arxiv.org/abs/2103.11922
- Code: https://github.com/xiusu/NAS-Bench-Macro

**Contrastive Neural Architecture Search with Neural Architecture Comparators**

- Paper: https://arxiv.org/abs/2103.05471
- Code: https://github.com/chenyaofo/CTNAS

**AttentiveNAS: Improving Neural Architecture Search via Attentive** 

- Paper: https://arxiv.org/abs/2011.09011
- Code: None

**ReNAS: Relativistic Evaluation of Neural Architecture Search**

- Paper: https://arxiv.org/abs/1910.01523
- Code: None

**HourNAS: Extremely Fast Neural Architecture**

- Paper: https://arxiv.org/abs/2005.14446
- Code: None

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

**TediGAN: Text-Guided Diverse Image Generation and Manipulation**

- Homepage: https://xiaweihao.com/projects/tedigan/

- Paper: https://arxiv.org/abs/2012.03308
- Code: https://github.com/weihaox/TediGAN

**Generative Hierarchical Features from Synthesizing Image**

- Homepage: https://genforce.github.io/ghfeat/

- Paper(Oral): https://arxiv.org/abs/2007.10379
- Code: https://github.com/genforce/ghfeat

**Teachers Do More Than Teach: Compressing Image-to-Image Models**

- Paper: https://arxiv.org/abs/2103.03467
- Code: https://github.com/snap-research/CAT

**HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms**

- Paper: https://arxiv.org/abs/2011.11731
- Code: https://github.com/mahmoudnafifi/HistoGAN

**pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis**

- Homepage: https://marcoamonteiro.github.io/pi-GAN-website/

- Paper(Oral): https://arxiv.org/abs/2012.00926
- Code: None

**DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network**

- Paper: https://arxiv.org/abs/2103.07893
- Code: None

**Diverse Semantic Image Synthesis via Probability Distribution Modeling**

- Paper: https://arxiv.org/abs/2103.06878
- Code: https://github.com/tzt101/INADE.git

**LOHO: Latent Optimization of Hairstyles via Orthogonalization**

- Paper: https://arxiv.org/abs/2103.03891
- Code: None

**PISE: Person Image Synthesis and Editing with Decoupled GAN**

- Paper: https://arxiv.org/abs/2103.04023
- Code: https://github.com/Zhangjinso/PISE

**DeFLOCNet: Deep Image Editing via Flexible Low-level Controls**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**PD-GAN: Probabilistic Diverse GAN for Image Inpainting**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**Efficient Conditional GAN Transfer with Knowledge Propagation across Classes**

- Paper: https://www.researchgate.net/publication/349309756_Efficient_Conditional_GAN_Transfer_with_Knowledge_Propagation_across_Classes
- Code: http://github.com/mshahbazi72/cGANTransfer

**Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**

- Paper: None
- Code: None

**Hijack-GAN: Unintended-Use of Pretrained, Black-Box GANs**

- Paper: https://arxiv.org/abs/2011.14107
- Code: None

**Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation**

- Homepage: https://eladrich.github.io/pixel2style2pixel/
- Paper: https://arxiv.org/abs/2008.00951
- Code: https://github.com/eladrich/pixel2style2pixel

**A 3D GAN for Improved Large-pose Facial Recognition**

- Paper: https://arxiv.org/abs/2012.10545
- Code: None

**HumanGAN: A Generative Model of Humans Images**

- Paper: https://arxiv.org/abs/2103.06902
- Code: None

**ID-Unet: Iterative Soft and Hard Deformation for View Synthesis**

- Paper: https://arxiv.org/abs/2103.02264
- Code: https://github.com/MingyuY/Iterative-view-synthesis

**CoMoGAN: continuous model-guided image-to-image translation**

- Paper(Oral): https://arxiv.org/abs/2103.06879
- Code: https://github.com/cv-rits/CoMoGAN

**Training Generative Adversarial Networks in One Stage**

- Paper: https://arxiv.org/abs/2103.00430
- Code: None

**Closed-Form Factorization of Latent Semantics in GANs**

- Homepage: https://genforce.github.io/sefa/
- Paper(Oral): https://arxiv.org/abs/2007.06600
- Code: https://github.com/genforce/sefa

**Anycost GANs for Interactive Image Synthesis and Editing**

- Paper: https://arxiv.org/abs/2103.03243
- Code: https://github.com/mit-han-lab/anycost-gan

**Image-to-image Translation via Hierarchical Style Disentanglement**

- Paper: https://arxiv.org/abs/2103.01456
- Code: https://github.com/imlixinyang/HiSD

<a name="VAE"></a>

# VAE

**Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders**

- Homepage: https://taldatech.github.io/soft-intro-vae-web/

- Paper: https://arxiv.org/abs/2012.13253
- Code: https://github.com/taldatech/soft-intro-vae-pytorch

<a name="Visual Transformer"></a>

# Visual Transformer

**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**

- Paper: https://arxiv.org/abs/2012.15840
- Code: https://github.com/fudan-zvg/SETR

**Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**

- Paper: https://arxiv.org/abs/2103.16553
- Code: None

**Transformer Tracking**

- Paper: https://arxiv.org/abs/2103.15436
- Code: https://github.com/chenxin-dlut/TransT

**HR-NAS: Searching Efficient High-Resolution Neural Architectures with Transformers**

- Paper(Oral): None
- Code: https://github.com/dingmyu/HR-NAS

**MIST: Multiple Instance Spatial Transformer Network**

- Paper: https://arxiv.org/abs/1811.10725
- Code: None

**Multimodal Motion Prediction with Stacked Transformers**

- Paper: https://arxiv.org/abs/2103.11624
- Code: https://decisionforce.github.io/mmTransformer

**Revamping cross-modal recipe retrieval with hierarchical Transformers and self-supervised learning**

- Paper: https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning

- Code: https://github.com/amzn/image-to-recipe-transformers

**Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking**

- Paper(Oral): https://arxiv.org/abs/2103.11681

- Code: https://github.com/594422814/TransformerTrack

**Pre-Trained Image Processing Transformer**

- Paper:  https://arxiv.org/abs/2012.00364 
- Code: None

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

<a name="SLAM"></a>

# SLAM

**Generalizing to the Open World: Deep Visual Odometry with Online Adaptation**

- Paper: https://arxiv.org/abs/2103.15279
- Code: https://arxiv.org/abs/2103.15279

<a name="Long-Tailed"></a>

# 长尾分布(Long-Tailed)

**Contrastive Learning based Hybrid Networks for Long-Tailed Image Classification**

- Paper: https://arxiv.org/abs/2103.14267
- Code: None

<a name="DA"></a>

# 数据增广(Data Augmentation)

**Scale-aware Automatic Augmentation for Object Detection**

- Paper: https://arxiv.org/abs/2103.17220

- Code: https://github.com/Jia-Research-Lab/SA-AutoAug

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

**Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework**

- Paper: https://arxiv.org/abs/2103.11402
- Code: None

**Adaptive Consistency Regularization for Semi-Supervised Transfer Learning**

- Paper: https://arxiv.org/abs/2103.02193
- Code: https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning

<a name="Capsule-Network"></a>

# 胶囊网络(Capsule Network)

**Capsule Network is Not More Robust than Convolutional Network**

- Paper: https://arxiv.org/abs/2103.15459
- Code: None

<a name="Object-Detection"></a>

# 2D目标检测(Object Detection)

## 2D目标检测

**Scale-aware Automatic Augmentation for Object Detection**

- Paper: https://arxiv.org/abs/2103.17220

- Code: https://github.com/Jia-Research-Lab/SA-AutoAug

**OTA: Optimal Transport Assignment for Object Detection**

- Paper: https://arxiv.org/abs/2103.14259
- Code: https://github.com/Megvii-BaseDetection/OTA

**Distilling Object Detectors via Decoupled Features**

- Paper: https://arxiv.org/abs/2103.14475
- Code: https://github.com/ggjy/DeFeat.pytorch

**Sparse R-CNN: End-to-End Object Detection with Learnable Proposals**

- Paper: https://arxiv.org/abs/2011.12450
- Code: https://github.com/PeizeSun/SparseR-CNN

**There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge**

- Homepage: https://rl.uni-freiburg.de/
- Paper: https://arxiv.org/abs/2103.01353
- Code: None

**Positive-Unlabeled Data Purification in the Wild for Object Detection**

- Paper: None
- Code: None

**Instance Localization for Self-supervised Detection Pretraining**

- Paper: https://arxiv.org/abs/2102.08318
- Code: https://github.com/limbo0000/InstanceLoc

**MeGA-CDA: Memory Guided Attention for Category-Aware Unsupervised Domain Adaptive Object Detection**

- Paper: https://arxiv.org/abs/2103.04224
- Code: None

**End-to-End Object Detection with Fully Convolutional Network**

- Paper: https://arxiv.org/abs/2012.03544
- Code: https://github.com/Megvii-BaseDetection/DeFCN

**Robust and Accurate Object Detection via Adversarial Learning**

- Paper: https://arxiv.org/abs/2103.13886

- Code: None

**I^3Net: Implicit Instance-Invariant Network for Adapting One-Stage Object Detectors**

- Paper: https://arxiv.org/abs/2103.13757
- Code: None 

**Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework**

- Paper: https://arxiv.org/abs/2103.11402
- Code: None

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

- Paper(Oral): https://arxiv.org/abs/2103.02603
- Code: https://github.com/JosephKJ/OWOD

## Few-Shot目标检测

**Semantic Relation Reasoning for Shot-Stable Few-Shot Object Detection**

- Paper: https://arxiv.org/abs/2103.01903
- Code: None

**Few-Shot Object Detection via Contrastive Proposal Encoding**

- Paper: https://arxiv.org/abs/2103.05950
- Code: https://github.com/MegviiDetection/FSCE 

## 旋转目标检测

**ReDet: A Rotation-equivariant Detector for Aerial Object Detection**

- Paper: https://arxiv.org/abs/2103.07733

- Code: https://github.com/csuhan/ReDet

<a name="Object-Tracking"></a>

# 单/多目标跟踪(Object Tracking)

## 单目标跟踪

**IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking**

- Paper: https://arxiv.org/abs/2103.14938
- Code: https://github.com/VISION-SJTU/IoUattack

**Graph Attention Tracking**

- Paper: https://arxiv.org/abs/2011.11204
- Code: https://github.com/ohhhyeahhh/SiamGAT

**Rotation Equivariant Siamese Networks for Tracking**

- Paper: https://arxiv.org/abs/2012.13078
- Code: None

**Track to Detect and Segment: An Online Multi-Object Tracker**

- Homepage: https://jialianwu.com/projects/TraDeS.html
- Paper: None
- Code: None

**Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking**

- Paper(Oral): https://arxiv.org/abs/2103.11681

- Code: https://github.com/594422814/TransformerTrack

**Transformer Tracking**

- Paper: https://arxiv.org/abs/2103.15436
- Code: https://github.com/chenxin-dlut/TransT

## 多目标跟踪

**Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking**

- Paper: https://arxiv.org/abs/2012.02337
- Code: None

**Learning a Proposal Classifier for Multiple Object Tracking**

- Paper: https://arxiv.org/abs/2103.07889
- Code: https://github.com/daip13/LPC_MOT.git

**Track to Detect and Segment: An Online Multi-Object Tracker**

- Homepage: https://jialianwu.com/projects/TraDeS.html
- Paper: https://arxiv.org/abs/2103.08808
- Code: https://github.com/JialianW/TraDeS

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers**

- Paper: https://arxiv.org/abs/2012.15840
- Code: https://github.com/fudan-zvg/SETR

**Bidirectional Projection Network for Cross Dimension Scene Understanding**

- Paper(Oral): https://arxiv.org/abs/2103.14326
- Code: https://github.com/wbhu/BPNet

**Cross-Dataset Collaborative Learning for Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.11351
- Code: None

**Continual Semantic Segmentation via Repulsion-Attraction of Sparse and Disentangled Latent Representations**

- Paper: https://arxiv.org/abs/2103.06342
- Code: None

**Capturing Omni-Range Context for Omnidirectional Segmentation**

- Paper: https://arxiv.org/abs/2103.05687
- Code: None

**Learning Statistical Texture for Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.04133
- Code: None

**PLOP: Learning without Forgetting for Continual Semantic Segmentation**

- Paper: https://arxiv.org/abs/2011.11390
- Code: None

## 弱监督语义分割

**Non-Salient Region Object Mining for Weakly Supervised Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.14581
- Code: None

**BBAM: Bounding Box Attribution Map for Weakly Supervised Semantic and Instance Segmentation**

- Paper: https://arxiv.org/abs/2103.08907
- Code: None

## 半监督语义分割

**Semi-supervised Domain Adaptation based on Dual-level Domain Mixing for Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.04705

## 域自适应语义分割

**RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening**

- Paper: https://arxiv.org/abs/2103.15597
- Code: https://github.com/shachoi/RobustNet

**Coarse-to-Fine Domain Adaptive Semantic Segmentation with Photometric Alignment and Category-Center Regularization**

- Paper: https://arxiv.org/abs/2103.13041
- Code: None

**MetaCorrection: Domain-aware Meta Loss Correction for Unsupervised Domain Adaptation in Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.05254
- Code: None

**Multi-Source Domain Adaptation with Collaborative Learning for Semantic Segmentation**

- Paper: https://arxiv.org/abs/2103.04717
- Code: None

**Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation**

- Paper: https://arxiv.org/abs/2101.10979
- Code: https://github.com/microsoft/ProDA

<a name="Instance-Segmentation"></a>

# 实例分割(Instance Segmentation)

**Boundary IoU: Improving Object-Centric Image Segmentation Evaluation**

- Homepage: https://bowenc0221.github.io/boundary-iou/
- Paper: https://arxiv.org/abs/2103.16562

- Code: https://github.com/bowenc0221/boundary-iou-api

**Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers**

- Paper: https://arxiv.org/abs/2103.12340

- Code: https://github.com/lkeab/BCNet 

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

## 3D医学图像分割

**DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation**

- Paper(Oral): https://arxiv.org/abs/2103.15954
- Code: None

<a name="IVOS"></a>

# 交互式视频目标分割(Interactive-Video-Object-Segmentation)

**Learning to Recommend Frame for Interactive Video Object Segmentation in the Wild**

- Paper: https://arxiv.org/abs/2103.10391

- Code: https://github.com/svip-lab/IVOS-W

<a name="Saliency-Detection"></a>

# 显著性检测(Saliency Detection)

**Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion**

- Paper(Oral): https://arxiv.org/abs/2103.11832
- Code: https://github.com/sunpeng1996/DSA2F

<a name="Person-Search"></a>

# 行人搜索(Person Search)

**Anchor-Free Person Search**

- Paper: https://arxiv.org/abs/2103.11617
- Code: https://github.com/daodaofr/AlignPS
- Interpretation: [首个无需锚框（Anchor-Free）的行人搜索框架 | CVPR 2021](https://mp.weixin.qq.com/s/iqJkgp0JBanmeBPyHUkb-A)

<a name="Video-Understanding"></a>

# 视频理解/行为识别(Video Understanding)

**No frame left behind: Full Video Action Recognition**

- Paper: https://arxiv.org/abs/2103.15395
- Code: None

**Learning Salient Boundary Feature for Anchor-free Temporal Action Localization**

- Paper: https://arxiv.org/abs/2103.13137
- Code: None

**Temporal Context Aggregation Network for Temporal Action Proposal Refinement**

- Paper: https://arxiv.org/abs/2103.13141
- Code: None
- Interpretation: [CVPR 2021 | TCANet：最强时序动作提名修正网络](https://mp.weixin.qq.com/s/UOWMfpTljkyZznHtpkQBhA)

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

**A 3D GAN for Improved Large-pose Facial Recognition**

- Paper: https://arxiv.org/abs/2012.10545
- Code: None

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

<a name="FER"></a>

# 人脸表情识别(Facial Expression Recognition)

**Affective Processes: stochastic modelling of temporal context for emotion and facial expression recognition**

- Paper: https://arxiv.org/abs/2103.13372
- Code: None

<a name="Deepfakes"></a>

# Deepfakes

**MagDR: Mask-guided Detection and Reconstruction for Defending Deepfakes**

- Paper: https://arxiv.org/abs/2103.14211
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

<a name="Animal-Pose-Estimation"></a>

# 动物姿态估计(Animal Pose Estimation)

**From Synthetic to Real: Unsupervised Domain Adaptation for Animal Pose Estimation**

- Paper: https://arxiv.org/abs/2103.14843
- Code: None

<a name="Human-Volumetric-Capture"></a>

# Human Volumetric Capture

**POSEFusion: Pose-guided Selective Fusion for Single-view Human Volumetric Capture**

- Homepage: http://www.liuyebin.com/posefusion/posefusion.html

- Paper(Oral): https://arxiv.org/abs/2103.15331
- Code: None

<a name="Scene-Text-Recognition"></a>

# 场景文本识别(Scene Text Recognition)

**Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition**

- Paper: https://arxiv.org/abs/2103.06495
- Code: https://github.com/FangShancheng/ABINet

<a name="Image-Compression"></a>

# 图像压缩

**Checkerboard Context Model for Efficient Learned Image Compression**

- Paper: https://arxiv.org/abs/2103.15306
- Code: None

**Slimmable Compressive Autoencoders for Practical Neural Image Compression**

- Paper: https://arxiv.org/abs/2103.15726
- Code: None

**Attention-guided Image Compression by Deep Reconstruction of Compressive Sensed Saliency Skeleton**

- Paper: https://arxiv.org/abs/2103.15368
- Code: None

<a name="Model-Compression"></a>

# 模型压缩/剪枝/量化

**Teachers Do More Than Teach: Compressing Image-to-Image Models**

- Paper: https://arxiv.org/abs/2103.03467
- Code: https://github.com/snap-research/CAT

## 模型剪枝

**Dynamic Slimmable Network**

- Paper: https://arxiv.org/abs/2103.13258
- Code: https://github.com/changlin31/DS-Net

## 模型量化

**Zero-shot Adversarial Quantization**

- Paper(Oral): https://arxiv.org/abs/2103.15263
- Code: https://git.io/Jqc0y

**Learnable Companding Quantization for Accurate Low-bit Neural Networks**

- Paper: https://arxiv.org/abs/2103.07156
- Code: None

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Distilling Object Detectors via Decoupled Features**

- Paper: https://arxiv.org/abs/2103.14475
- Code: https://github.com/ggjy/DeFeat.pytorch

<a name="Super-Resolution"></a>

# 超分辨率(Super-Resolution)

**ClassSR: A General Framework to Accelerate Super-Resolution Networks by Data Characteristic**

- Paper: https://arxiv.org/abs/2103.04039
- Code: https://github.com/Xiangtaokong/ClassSR

**AdderSR: Towards Energy Efficient Image Super-Resolution**

- Paper: https://arxiv.org/abs/2009.08891
- Code: None

## 视频超分辨率

**Temporal Modulation Network for Controllable Space-Time Video Super-Resolution**

- Paper: None
- Code: https://github.com/CS-GangXu/TMNet

<a name="Image-Restoration"></a>

# 图像恢复(Image Restoration)

**Multi-Stage Progressive Image Restoration**

- Paper: https://arxiv.org/abs/2102.02808
- Code: https://github.com/swz30/MPRNet

<a name="Image-Inpainting"></a>

# 图像补全(Image Inpainting)

**TransFill: Reference-guided Image Inpainting by Merging Multiple Color and Spatial Transformations**

- Homepage: https://yzhouas.github.io/projects/TransFill/index.html
- Paper: https://arxiv.org/abs/2103.15982
- Code: None

**PD-GAN: Probabilistic Diverse GAN for Image Inpainting**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**High-Fidelity and Arbitrary Face Editing**

- Paper: https://arxiv.org/abs/2103.15814
- Code: None

**Anycost GANs for Interactive Image Synthesis and Editing**

- Paper: https://arxiv.org/abs/2103.03243
- Code: https://github.com/mit-han-lab/anycost-gan

**PISE: Person Image Synthesis and Editing with Decoupled GAN**

- Paper: https://arxiv.org/abs/2103.04023
- Code: https://github.com/Zhangjinso/PISE

**DeFLOCNet: Deep Image Editing via Flexible Low-level Controls**

- Paper: http://raywzy.com/
- Code: http://raywzy.com/

**Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing**

- Paper: None
- Code: None

<a name="Image-Matching"></a>

# 图像匹配(Image Matcing)

**Convolutional Hough Matching Networks**

- Homapage: http://cvlab.postech.ac.kr/research/CHM/
- Paper(Oral): https://arxiv.org/abs/2103.16831
- Code: None

<a name="Reflection-Removal"></a>

# 反光去除(Reflection Removal)

**Robust Reflection Removal with Reflection-free Flash-only Cues**

- Paper: https://arxiv.org/abs/2103.04273
- Code: https://github.com/ChenyangLEI/flash-reflection-removal

<a name="3D-C"></a>

# 3D点云分类(3D Point Clouds Classification)

**Equivariant Point Network for 3D Point Cloud Analysis**

- Paper: https://arxiv.org/abs/2103.14147
- Code: None

**PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds**

- Paper: https://arxiv.org/abs/2103.14635
- Code: https://github.com/CVMI-Lab/PAConv

<a name="3D-Object-Detection"></a>

# 3D目标检测(3D Object Detection)

**LiDAR R-CNN: An Efficient and Universal 3D Object Detector**

- Paper: https://arxiv.org/abs/2103.15297
- Code: https://github.com/tusimple/LiDAR_RCNN

**M3DSSD: Monocular 3D Single Stage Object Detector**

- Paper: https://arxiv.org/abs/2103.13164

- Code: https://github.com/mumianyuxin/M3DSSD

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

**Bidirectional Projection Network for Cross Dimension Scene Understanding**

- Paper(Oral): https://arxiv.org/abs/2103.14326
- Code: https://github.com/wbhu/BPNet

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

<a name="3D-Panoptic-Segmentation"></a>

# 3D全景分割(3D Panoptic Segmentation)

**Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation**

- Paper: https://arxiv.org/abs/2103.14962
- Code: None

<a name="3D-Object-Tracking"></a>

# 3D目标跟踪(3D Object Trancking)

**Center-based 3D Object Detection and Tracking**

- Paper: https://arxiv.org/abs/2006.11275
- Code: https://github.com/tianweiy/CenterPoint

<a name="3D-PointCloud-Registration"></a>

# 3D点云配准(3D Point Cloud Registration)

**ReAgent: Point Cloud Registration using Imitation and Reinforcement Learning**

- Paper: https://arxiv.org/abs/2103.15231
- Code: None

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

**FS-Net: Fast Shape-based Network for Category-Level 6D Object Pose Estimation with Decoupled Rotation Mechanism**

- Paper(Oral): https://arxiv.org/abs/2103.07054
- Code: https://github.com/DC1991/FS-Net

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

**LiBRe: A Practical Bayesian Approach to Adversarial Detection**

- Paper: https://arxiv.org/abs/2103.14835
- Code: None

**Natural Adversarial Examples**

- Paper: https://arxiv.org/abs/1907.07174
- Code: https://github.com/hendrycks/natural-adv-examples

<a name="Image-Retrieval"></a>

# 图像检索(Image Retrieval)

**StyleMeUp: Towards Style-Agnostic Sketch-Based Image Retrieval**

- Paper: https://arxiv.org/abs/2103.15706
- COde: None

**QAIR: Practical Query-efficient Black-Box Attacks for Image Retrieval**

- Paper: https://arxiv.org/abs/2103.02927
- Code: None

<a name="Video-Retrieval"></a>

# 视频检索(Video Retrieval)

**On Semantic Similarity in Video Retrieval**

- Paper: https://arxiv.org/abs/2103.10095

- Homepage: https://mwray.github.io/SSVR/
- Code: https://github.com/mwray/Semantic-Video-Retrieval

<a name="Cross-modal-Retrieval"></a>

# 跨模态检索(Cross-modal Retrieval)

**Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers**

- Paper: https://arxiv.org/abs/2103.16553
- Code: None

**Revamping cross-modal recipe retrieval with hierarchical Transformers and self-supervised learning**

- Paper: https://www.amazon.science/publications/revamping-cross-modal-recipe-retrieval-with-hierarchical-transformers-and-self-supervised-learning

- Code: https://github.com/amzn/image-to-recipe-transformers

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

**CDFI: Compression-Driven Network Design for Frame Interpolation**

- Paper: None
- Code: https://github.com/tding1/CDFI

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

**Self-Supervised Visibility Learning for Novel View Synthesis**

- Paper: https://arxiv.org/abs/2103.15407
- Code: None

**NeX: Real-time View Synthesis with Neural Basis Expansion**

- Homepage: https://nex-mpi.github.io/
- Paper(Oral): https://arxiv.org/abs/2103.05606

<a name="Domain-Generalization"></a>

# Domain Generalization

**RobustNet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening**

- Paper: https://arxiv.org/abs/2103.15597
- Code: https://github.com/shachoi/RobustNet

**Adaptive Methods for Real-World Domain Generalization**

- Paper: https://arxiv.org/abs/2103.15796
- Code: None

**FSDR: Frequency Space Domain Randomization for Domain Generalization**

- Paper: https://arxiv.org/abs/2103.02370
- Code: None

<a name="Open-Set-Recognition"></a>

# Open-Set Recognition

**Learning Placeholders for Open-Set Recognition**

- Paper(Oral): https://arxiv.org/abs/2103.15086
- Code: None

<a name="Adversarial-Attack"></a>

# Adversarial Attack

**IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking**

- Paper: https://arxiv.org/abs/2103.14938
- Code: https://github.com/VISION-SJTU/IoUattack

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

**Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark**

- Homepage: https://vap.aau.dk/sewer-ml/
- Paper: https://arxiv.org/abs/2103.10619

**Sewer-ML: A Multi-Label Sewer Defect Classification Dataset and Benchmark**

- Homepage: https://vap.aau.dk/sewer-ml/

- Paper: https://arxiv.org/abs/2103.10895

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

**Tuning IR-cut Filter for Illumination-aware Spectral Reconstruction from RGB**

- Paper(Oral): https://arxiv.org/abs/2103.14708
- Code: None

**Invertible Image Signal Processing**

- Paper: https://arxiv.org/abs/2103.15061
- Code: https://github.com/yzxing87/Invertible-ISP

**Video Rescaling Networks with Joint Optimization Strategies for Downscaling and Upscaling**

- Paper: https://arxiv.org/abs/2103.14858
- Code: None

**SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences**

- Paper: https://arxiv.org/abs/2103.14898
- Code: None

**Embedding Transfer with Label Relaxation for Improved Metric Learning**

- Paper: https://arxiv.org/abs/2103.14908
- Code: None

**Picasso: A CUDA-based Library for Deep Learning over 3D Meshes**

- Paper: https://arxiv.org/abs/2103.15076 
- Code: https://github.com/hlei-ziyan/Picasso

**Meta-Mining Discriminative Samples for Kinship Verification**

- Paper: https://arxiv.org/abs/2103.15108
- Code: None

**Cloud2Curve: Generation and Vectorization of Parametric Sketches**

- Paper: https://arxiv.org/abs/2103.15536
- Code: None

**TrafficQA: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events**

- Paper: https://arxiv.org/abs/2103.15538
- Code: https://github.com/SUTDCV/SUTD-TrafficQA

**Abstract Spatial-Temporal Reasoning via Probabilistic Abduction and Execution**

- Homepage: http://wellyzhang.github.io/project/prae.html

- Paper: https://arxiv.org/abs/2103.14230
- Code: None

**ACRE: Abstract Causal REasoning Beyond Covariation**

- Homepage: http://wellyzhang.github.io/project/acre.html

- Paper: https://arxiv.org/abs/2103.14232
- Code: None

**Confluent Vessel Trees with Accurate Bifurcations**

- Paper: https://arxiv.org/abs/2103.14268
- Code: None

**Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modeling**

- Paper: https://arxiv.org/abs/2103.14338
- Code: https://github.com/HuangZhiChao95/FewShotMotionTransfer

**Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks**

- Homepage: https://paschalidoud.github.io/neural_parts
- Paper: None 
- Code: https://github.com/paschalidoud/neural_parts 

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

