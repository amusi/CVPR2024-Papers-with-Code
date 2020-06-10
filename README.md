# CVPR2020-Code

CVPR 2020 论文开源项目合集，同时欢迎各位大佬提交issue，分享CVPR 2020开源项目

- [CNN](#CNN)
- [图像分类](#Image-Classification)
- [目标检测](#Object-Detection)
- [3D目标检测](#3D-Object-Detection)
- [视频目标检测](#Video-Object-Detection)
- [目标跟踪](#Object-Tracking)
- [语义分割](#Semantic-Segmentation)
- [实例分割](#Instance-Segmentation)
- [全景分割](#Panoptic-Segmentation)
- [视频目标分割](#VOS)
- [超像素分割](#Superpixel)
- [NAS](#NAS)
- [GAN](#GAN)
- [Re-ID](#Re-ID)
- [3D点云（分类/分割/配准/跟踪等）](#3D-PointCloud)
- [人脸（识别/检测/重建等）](#Face)
- [人体姿态估计(2D/3D)](#Human-Pose-Estimation)
- [人体解析](#Human-Parsing)
- [场景文本检测](#Scene-Text-Detection)
- [场景文本识别](#Scene-Text-Recognition)
- [超分辨率](#Super-Resolution)
- [模型压缩/剪枝](#Model-Compression)
- [视频理解/行为识别](#Action-Recognition)
- [人群计数](#Crowd-Counting)
- [深度估计](#Depth-Estimation)
- [6D目标姿态估计](#6DOF)
- [手势估计](#Hand-Pose)
- [显著性检测](#Saliency)
- [去噪](#Denoising)
- [去模糊](#Deblurring)
- [去雾](#Dehazing)
- [特征点检测与描述](#Feature)
- [视觉问答(VQA)](#VQA)
- [视频问答(VideoQA)](#VideoQA)
- [视觉语言导航](#VLN)
- [视频压缩](#Video-Compression)
- [视频插值](#Video-Frame-Interpolation)
- [风格迁移](#Style-Transfer)
- [车道线检测](#Lane-Detection)
- ["人-物"交互(HOI)检测](#HOI)
- [行为轨迹预测](#HTP)
- [运动预测](#Motion-Predication)
- [虚拟试衣](#Virtual-Try-On)
- [HDR](#HDR)
- [对抗样本](#AE)
- [深度补全](#DC)
- [语义场景补全](#SSC)
- [数据集](#Datasets)
- [其他](#Others)
- [不确定中没中](#Not-Sure)

<a name="CNN"></a>

# CNN

**Exploring Self-attention for Image Recognition**

- 论文：https://hszhao.github.io/papers/cvpr20_san.pdf

- 代码：https://github.com/hszhao/SAN

**Improving Convolutional Networks with Self-Calibrated Convolutions**

- 主页：https://mmcheng.net/scconv/

- 论文：http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf
- 代码：https://github.com/backseason/SCNet

**Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**

- 论文：https://arxiv.org/abs/2003.13549
- 代码：https://github.com/zeiss-microscopy/BSConv

<a name="Image-Classification"></a>

# 图像分类

**Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion**

- 论文：https://arxiv.org/abs/2003.04490

- 代码：https://github.com/AdamKortylewski/CompositionalNets

**Spatially Attentive Output Layer for Image Classification**

- 论文：https://arxiv.org/abs/2004.07570 

- 代码（好像被原作者删除了）：https://github.com/ildoonet/spatially-attentive-output-layer 

<a name="Object-Detection"></a>

# 目标检测

**D2Det: Towards High Quality Object Detection and Instance Segmentation**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf

- 代码：https://github.com/JialeCao001/D2Det

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**

- 论文下载链接：https://arxiv.org/abs/2005.09973

- 代码和数据集：https://github.com/Anymake/DRN_CVPR2020

**Scale-Equalizing Pyramid Convolution for Object Detection**

论文：https://arxiv.org/abs/2005.03101

代码：https://github.com/jshilong/SEPC

**Revisiting the Sibling Head in Object Detector**

- 论文：https://arxiv.org/abs/2003.07540

- 代码：https://github.com/Sense-X/TSD 

**Scale-equalizing Pyramid Convolution for Object Detection**

- 论文：暂无
- 代码：https://github.com/jshilong/SEPC 

**Detection in Crowded Scenes: One Proposal, Multiple Predictions**

- 论文：https://arxiv.org/abs/2003.09163
- 代码：https://github.com/megvii-model/CrowdDetection

**Instance-aware, Context-focused, and Memory-efficient Weakly Supervised Object Detection**

- 论文：https://arxiv.org/abs/2004.04725
- 代码：https://github.com/NVlabs/wetectron

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**

- 论文：https://arxiv.org/abs/1912.02424 
- 代码：https://github.com/sfzhang15/ATSS

**BiDet: An Efficient Binarized Object Detector**

- 论文：https://arxiv.org/abs/2003.03961 
- 代码：https://github.com/ZiweiWangTHU/BiDet

**Harmonizing Transferability and Discriminability for Adapting Object Detectors**

- 论文：https://arxiv.org/abs/2003.06297
- 代码：https://github.com/chaoqichen/HTCN

**CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection**

- 论文：https://arxiv.org/abs/2003.09119
- 代码：https://github.com/KiveeDong/CentripetalNet

**Hit-Detector: Hierarchical Trinity Architecture Search for Object Detection**

- 论文：https://arxiv.org/abs/2003.11818
- 代码：https://github.com/ggjy/HitDet.pytorch

**EfficientDet: Scalable and Efficient Object Detection**

- 论文：https://arxiv.org/abs/1911.09070
- 代码：https://github.com/google/automl/tree/master/efficientdet 

<a name="3D-Object-Detection"></a>

# 3D目标检测

**Train in Germany, Test in The USA: Making 3D Object Detectors Generalize**

- 论文：https://arxiv.org/abs/2005.08139

- 代码：https://github.com/cxy1997/3D_adapt_auto_driving

**MLCVNet: Multi-Level Context VoteNet for 3D Object Detection**

- 论文：https://arxiv.org/abs/2004.05679
- 代码：https://github.com/NUAAXQ/MLCVNet

**3DSSD: Point-based 3D Single Stage Object Detector**

- CVPR 2020 Oral

- 论文：https://arxiv.org/abs/2002.10187

- 代码：https://github.com/tomztyang/3DSSD

**Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation**

- 论文：https://arxiv.org/abs/2004.03572

- 代码：https://github.com/zju3dv/disprcn

**End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection**

- 论文：https://arxiv.org/abs/2004.03080

- 代码：https://github.com/mileyan/pseudo-LiDAR_e2e

**DSGN: Deep Stereo Geometry Network for 3D Object Detection**

- 论文：https://arxiv.org/abs/2001.03398
- 代码：https://github.com/chenyilun95/DSGN

**LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention**

- 论文：https://arxiv.org/abs/2004.01389
- 代码：https://github.com/yinjunbo/3DVID

**PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection**

- 论文：https://arxiv.org/abs/1912.13192

- 代码：https://github.com/sshaoshuai/PV-RCNN

**Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud**

- 论文：https://arxiv.org/abs/2003.01251 
- 代码：https://github.com/WeijingShi/Point-GNN 

<a name="Video-Object-Detection"></a>

# 视频目标检测

**Memory Enhanced Global-Local Aggregation for Video Object Detection**

论文：https://arxiv.org/abs/2003.12063

代码：https://github.com/Scalsol/mega.pytorch

<a name="Object-Tracking"></a>

# 目标跟踪

**SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking**

- 论文：https://arxiv.org/abs/1911.07241
- 代码：https://github.com/ohhhyeahhh/SiamCAR

**D3S -- A Discriminative Single Shot Segmentation Tracker**

- 论文：https://arxiv.org/abs/1911.08862
- 代码：https://github.com/alanlukezic/d3s

**ROAM: Recurrently Optimizing Tracking Model**

- 论文：https://arxiv.org/abs/1907.12006

- 代码：https://github.com/skyoung/ROAM

**Siam R-CNN: Visual Tracking by Re-Detection**

- 主页：https://www.vision.rwth-aachen.de/page/siamrcnn
- 论文：https://arxiv.org/abs/1911.12836
- 论文2：https://www.vision.rwth-aachen.de/media/papers/192/siamrcnn.pdf
- 代码：https://github.com/VisualComputingInstitute/SiamR-CNN

**Cooling-Shrinking Attack: Blinding the Tracker with Imperceptible Noises**

- 论文：https://arxiv.org/abs/2003.09595 
- 代码：https://github.com/MasterBin-IIAU/CSA 

**High-Performance Long-Term Tracking with Meta-Updater**

- 论文：https://arxiv.org/abs/2004.00305

- 代码：https://github.com/Daikenan/LTMU

**AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization**

- 论文：https://arxiv.org/abs/2003.12949

- 代码：https://github.com/vision4robotics/AutoTrack

**Probabilistic Regression for Visual Tracking**

- 论文：https://arxiv.org/abs/2003.12565
- 代码：https://github.com/visionml/pytracking

**MAST: A Memory-Augmented Self-supervised Tracker**

- 论文：https://arxiv.org/abs/2002.07793
- 代码：https://github.com/zlai0/MAST

**Siamese Box Adaptive Network for Visual Tracking**

- 论文：https://arxiv.org/abs/2003.06761

- 代码：https://github.com/hqucv/siamban

<a name="Semantic-Segmentation"></a>

# 语义分割

**Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation**

- 论文：暂无

- 代码：https://github.com/JianqiangWan/Super-BPD

**Single-Stage Semantic Segmentation from Image Labels**

- 论文：https://arxiv.org/abs/2005.08104

- 代码：https://github.com/visinf/1-stage-wseg

**Learning Texture Invariant Representation for Domain Adaptation of Semantic Segmentation**

- 论文：https://arxiv.org/abs/2003.00867
- 代码：https://github.com/MyeongJin-Kim/Learning-Texture-Invariant-Representation

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- 论文：http://vladlen.info/papers/MSeg.pdf
- 代码：https://github.com/mseg-dataset/mseg-api

**CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement**

- 论文：https://arxiv.org/abs/2005.02551
- 代码：https://github.com/hkchengrex/CascadePSP

**Unsupervised Intra-domain Adaptation for Semantic Segmentation through Self-Supervision**

- Oral
- 论文：https://arxiv.org/abs/2004.07703
- 代码：https://github.com/feipan664/IntraDA

**Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**

- 论文：https://arxiv.org/abs/2004.04581
- 代码：https://github.com/YudeWang/SEAM

**Temporally Distributed Networks for Fast Video Segmentation**

- 论文：https://arxiv.org/abs/2004.01800

- 代码：https://github.com/feinanshan/TDNet

**Context Prior for Scene Segmentation**

- 论文：https://arxiv.org/abs/2004.01547

- 代码：https://git.io/ContextPrior

**Strip Pooling: Rethinking Spatial Pooling for Scene Parsing**

- 论文：https://arxiv.org/abs/2003.13328

- 代码：https://github.com/Andrew-Qibin/SPNet

**Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**

- 论文：https://arxiv.org/abs/2003.05128
- 代码：https://github.com/shachoi/HANet

**Learning Dynamic Routing for Semantic Segmentation**

- 论文：https://arxiv.org/abs/2003.10401

- 代码：https://github.com/yanwei-li/DynamicRouting

<a name="Instance-Segmentation"></a>

# 实例分割

**D2Det: Towards High Quality Object Detection and Instance Segmentation**

- 论文：http://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf

- 代码：https://github.com/JialeCao001/D2Det

**PolarMask: Single Shot Instance Segmentation with Polar Representation**

- 论文：https://arxiv.org/abs/1909.13226 
- 代码：https://github.com/xieenze/PolarMask 
- 解读：https://zhuanlan.zhihu.com/p/84890413 

**CenterMask : Real-Time Anchor-Free Instance Segmentation**

- 论文：https://arxiv.org/abs/1911.06667 
- 代码：https://github.com/youngwanLEE/CenterMask 

**BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation**

- 论文：https://arxiv.org/abs/2001.00309
- 代码：https://github.com/aim-uofa/AdelaiDet

**Deep Snake for Real-Time Instance Segmentation**

- 论文：https://arxiv.org/abs/2001.01629
- 代码：https://github.com/zju3dv/snake

**Mask Encoding for Single Shot Instance Segmentation**

- 论文：https://arxiv.org/abs/2003.11712

- 代码：https://github.com/aim-uofa/AdelaiDet

<a name="Panoptic-Segmentation"></a>

# 全景分割

**Pixel Consensus Voting for Panoptic Segmentation**

- 论文：https://arxiv.org/abs/2004.01849
- 代码：还未公布

**BANet: Bidirectional Aggregation Network with Occlusion Handling for Panoptic Segmentation**

论文：https://arxiv.org/abs/2003.14031

代码：https://github.com/Mooonside/BANet

<a name="VOS"></a>

# 视频目标分割

**A Transductive Approach for Video Object Segmentation**

- 论文：https://arxiv.org/abs/2004.07193

- 代码：https://github.com/microsoft/transductive-vos.pytorch

**State-Aware Tracker for Real-Time Video Object Segmentation**

- 论文：https://arxiv.org/abs/2003.00482

- 代码：https://github.com/MegviiDetection/video_analyst

**Learning Fast and Robust Target Models for Video Object Segmentation**

- 论文：https://arxiv.org/abs/2003.00908 
- 代码：https://github.com/andr345/frtm-vos

**Learning Video Object Segmentation from Unlabeled Videos**

- 论文：https://arxiv.org/abs/2003.05020
- 代码：https://github.com/carrierlxk/MuG

<a name="Superpixel"></a>

# 超像素分割

**Superpixel Segmentation with Fully Convolutional Networks**

- 论文：https://arxiv.org/abs/2003.12929
- 代码：https://github.com/fuy34/superpixel_fcn

<a name="NAS"></a>

# NAS

**AOWS: Adaptive and optimal network width search with latency constraints**

- 论文：https://arxiv.org/abs/2005.10481
- 代码：https://github.com/bermanmaxim/AOWS

**Densely Connected Search Space for More Flexible Neural Architecture Search**

- 论文：https://arxiv.org/abs/1906.09607

- 代码：https://github.com/JaminFong/DenseNAS

**MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning**

- 论文：https://arxiv.org/abs/2003.14058

- 代码：https://github.com/bhpfelix/MTLNAS

**FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions**

- 论文下载链接：https://arxiv.org/abs/2004.05565

- 代码：https://github.com/facebookresearch/mobile-vision

**Neural Architecture Search for Lightweight Non-Local Networks**

- 论文：https://arxiv.org/abs/2004.01961
- 代码：https://github.com/LiYingwei/AutoNL

**Rethinking Performance Estimation in Neural Architecture Search**

- 论文：https://arxiv.org/abs/2005.09917
- 代码：https://github.com/zhengxiawu/rethinking_performance_estimation_in_NAS
- 解读1：https://www.zhihu.com/question/372070853/answer/1035234510
- 解读2：https://zhuanlan.zhihu.com/p/111167409

**CARS: Continuous Evolution for Efficient Neural Architecture Search**

- 论文：https://arxiv.org/abs/1909.04977 
- 代码（即将开源）：https://github.com/huawei-noah/CARS 

<a name="GAN"></a>

# GAN

**Semantically Mutil-modal Image Synthesis**

- 主页：http://seanseattle.github.io/SMIS
- 论文：https://arxiv.org/abs/2003.12697
- 代码：https://github.com/Seanseattle/SMIS

**Unpaired Portrait Drawing Generation via Asymmetric Cycle Mapping**

- 论文：https://yiranran.github.io/files/CVPR2020_Unpaired%20Portrait%20Drawing%20Generation%20via%20Asymmetric%20Cycle%20Mapping.pdf
- 代码：https://github.com/yiranran/Unpaired-Portrait-Drawing

**Learning to Cartoonize Using White-box Cartoon Representations**

- 论文：https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/paper/06791.pdf

- 主页：https://systemerrorwang.github.io/White-box-Cartoonization/
- 代码：https://github.com/SystemErrorWang/White-box-Cartoonization
- 解读：https://zhuanlan.zhihu.com/p/117422157
- Demo视频：https://www.bilibili.com/video/av56708333

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 论文：https://arxiv.org/abs/2003.08936

- 代码：https://github.com/mit-han-lab/gan-compression

**Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**

- 论文：https://arxiv.org/abs/2003.01826 
- 代码：https://github.com/cc-hpc-itwm/UpConv 

<a name="Re-ID"></a>

# Re-ID

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**

- 论文：https://arxiv.org/abs/2005.07862

- 数据集：暂无

**Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking**

- 论文：https://arxiv.org/abs/2004.04199

- 代码：https://github.com/whj363636/Adversarial-attack-on-Person-ReID-With-Deep-Mis-Ranking

**Pose-guided Visible Part Matching for Occluded Person ReID**

- 论文：https://arxiv.org/abs/2004.00230
- 代码：https://github.com/hh23333/PVPM

**Weakly supervised discriminative feature learning with state information for person identification**

- 论文：https://arxiv.org/abs/2002.11939 
- 代码：https://github.com/KovenYu/state-information 

<a name="3D-PointCloud"></a>

# 3D点云（分类/分割/配准等）

## 3D点云卷积

**Global-Local Bidirectional Reasoning for Unsupervised Representation Learning of 3D Point Clouds**

- 论文下载链接：https://arxiv.org/abs/2003.12971

- 代码：https://github.com/raoyongming/PointGLR

**Grid-GCN for Fast and Scalable Point Cloud Learning**

- 论文：https://arxiv.org/abs/1912.02984

- 代码：https://github.com/Xharlie/Grid-GCN

**FPConv: Learning Local Flattening for Point Convolution**

- 论文：https://arxiv.org/abs/2002.10701
- 代码：https://github.com/lyqun/FPConv

## 3D点云分类

**PointAugment: an Auto-Augmentation Framework for Point Cloud Classification**

- 论文：https://arxiv.org/abs/2002.10876 
- 代码（即将开源）： https://github.com/liruihui/PointAugment/ 

## 3D点云语义分割

**RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**

- 论文：https://arxiv.org/abs/1911.11236
- 代码：https://github.com/QingyongHu/RandLA-Net

- 解读：https://zhuanlan.zhihu.com/p/105433460

**Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels**

- 论文：https://arxiv.org/abs/2004.0409

- 代码：https://github.com/alex-xun-xu/WeakSupPointCloudSeg

**PolarNet: An Improved Grid Representation for Online LiDAR Point Clouds Semantic Segmentation**

- 论文：https://arxiv.org/abs/2003.14032
- 代码：https://github.com/edwardzhou130/PolarSeg

**Learning to Segment 3D Point Clouds in 2D Image Space**

- 论文：https://arxiv.org/abs/2003.05593

- 代码：https://github.com/WPI-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space

## 3D点云实例分割

PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation

- 论文：https://arxiv.org/abs/2004.01658
- 代码：https://github.com/Jia-Research-Lab/PointGroup

## 3D点云配准

**D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features**

- 论文：https://arxiv.org/abs/2003.03164
- 代码：https://github.com/XuyangBai/D3Feat

**RPM-Net: Robust Point Matching using Learned Features**

- 论文：https://arxiv.org/abs/2003.13479
- 代码：https://github.com/yewzijian/RPMNet 

## 3D点云补全

**Cascaded Refinement Network for Point Cloud Completion**

- 论文：https://arxiv.org/abs/2004.03327
- 代码：https://github.com/xiaogangw/cascaded-point-completion

## 3D点云目标跟踪

**P2B: Point-to-Box Network for 3D Object Tracking in Point Clouds**

- 论文：https://arxiv.org/abs/2005.13888
- 代码：https://github.com/HaozheQi/P2B

<a name="Face"></a>

# 人脸

## 人脸识别

**CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition**

- 论文：https://arxiv.org/abs/2004.00288

- 代码：https://github.com/HuangYG123/CurricularFace

**Learning Meta Face Recognition in Unseen Domains**

- 论文：https://arxiv.org/abs/2003.07733
- 代码：https://github.com/cleardusk/MFR
- 解读：https://mp.weixin.qq.com/s/YZoEnjpnlvb90qSI3xdJqQ 

## 人脸检测

## 人脸活体检测

**Searching Central Difference Convolutional Networks for Face Anti-Spoofing**

- 论文：https://arxiv.org/abs/2003.04092

- 代码：https://github.com/ZitongYu/CDCN

## 人脸表情识别

**Suppressing Uncertainties for Large-Scale Facial Expression Recognition**

- 论文：https://arxiv.org/abs/2002.10392 

- 代码（即将开源）：https://github.com/kaiwang960112/Self-Cure-Network 

## 人脸转正

**Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images**

- 论文：https://arxiv.org/abs/2003.08124
- 代码：https://github.com/Hangz-nju-cuhk/Rotate-and-Render

## 人脸3D重建

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- 论文：https://arxiv.org/abs/2003.13845
- 数据集：https://github.com/lattas/AvatarMe

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- 论文：https://arxiv.org/abs/2003.13989
- 代码：https://github.com/zhuhao-nju/facescape

<a name="Human-Pose-Estimation"></a>

# 人体姿态估计(2D/3D)

## 2D人体姿态估计

**HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation**

- 论文：https://arxiv.org/abs/1908.10357
- 代码：https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

**The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation**

- 论文：https://arxiv.org/abs/1911.07524 
- 代码：https://github.com/HuangJunJie2017/UDP-Pose
- 解读：https://zhuanlan.zhihu.com/p/92525039

**Distribution-Aware Coordinate Representation for Human Pose Estimation**

- 主页：https://ilovepose.github.io/coco/ 

- 论文：https://arxiv.org/abs/1910.06278 

- 代码：https://github.com/ilovepose/DarkPose 

## 3D人体姿态估计

**Fusing Wearable IMUs with Multi-View Images for Human Pose Estimation: A Geometric Approach**

- 主页：https://www.zhe-zhang.com/cvpr2020
- 论文：https://arxiv.org/abs/2003.11163

- 代码：https://github.com/CHUNYUWANG/imu-human-pose-pytorch

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- 论文下载链接：https://arxiv.org/abs/2004.01166

- 代码：https://github.com/Healthcare-Robotics/bodies-at-rest
- 数据集：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis**

- 主页：http://val.cds.iisc.ac.in/pgp-human/
- 论文：https://arxiv.org/abs/2004.04400

**Compressed Volumetric Heatmaps for Multi-Person 3D Pose Estimation**

- 论文：https://arxiv.org/abs/2004.00329
- 代码：https://github.com/fabbrimatteo/LoCO

**VIBE: Video Inference for Human Body Pose and Shape Estimation**

- 论文：https://arxiv.org/abs/1912.05656 
- 代码：https://github.com/mkocabas/VIBE

**Back to the Future: Joint Aware Temporal Deep Learning 3D Human Pose Estimation**

- 论文：https://arxiv.org/abs/2002.11251 
- 代码：https://github.com/vnmr/JointVideoPose3D

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 论文：https://arxiv.org/abs/2003.03972
- 数据集：暂无

<a name="Human-Parsing"></a>

# 人体解析

**Correlating Edge, Pose with Parsing**

- 论文：https://arxiv.org/abs/2005.01431

- 代码：https://github.com/ziwei-zh/CorrPM

<a name="Scene-Text-Detection"></a>

# 场景文本检测

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 论文：https://arxiv.org/abs/2002.10200 
- 代码（即将开源）：https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- 代码（即将开源）：https://github.com/aim-uofa/adet

**Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection**

- 论文：https://arxiv.org/abs/2003.07493

- 代码：https://github.com/GXYM/DRRG

<a name="Scene-Text-Recognition"></a>

# 场景文本识别

**SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition**

- 论文：https://arxiv.org/abs/2005.10977
- 代码：https://github.com/Pay20Y/SEED

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**ABCNet: Real-time Scene Text Spotting with Adaptive Bezier-Curve Network**

- 论文：https://arxiv.org/abs/2002.10200 
- 代码（即将开源）：https://github.com/aim-uofa/adet

**Learn to Augment: Joint Data Augmentation and Network Optimization for Text Recognition**

- 论文：https://arxiv.org/abs/2003.06606

- 代码：https://github.com/Canjie-Luo/Text-Image-Augmentation

<a name="Super-Resolution"></a>

# 超分辨率

## 图像超分辨率

**Learning Texture Transformer Network for Image Super-Resolution**

- 论文：https://arxiv.org/abs/2006.04139

- 代码：https://github.com/FuzhiYang/TTSR

**Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining**

- 论文：https://arxiv.org/abs/2006.01424
- 代码：https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention

**Structure-Preserving Super Resolution with Gradient Guidance**

- 论文：https://arxiv.org/abs/2003.13081

- 代码：https://github.com/Maclory/SPSR

**Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy**

论文：https://arxiv.org/abs/2004.00448

代码：https://github.com/clovaai/cutblur

## 视频超分辨率

**Space-Time-Aware Multi-Resolution Video Enhancement**

- 主页：https://alterzero.github.io/projects/STAR.html
- 论文：http://arxiv.org/abs/2003.13170
- 代码：https://github.com/alterzero/STARnet

**Zooming Slow-Mo: Fast and Accurate One-Stage Space-Time Video Super-Resolution**

- 论文：https://arxiv.org/abs/2002.11616 
- 代码：https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020 

<a name="Model-Compression"></a>

# 模型压缩/剪枝

**DMCP: Differentiable Markov Channel Pruning for Neural Networks**

- 论文：https://arxiv.org/abs/2005.03354
- 代码：https://github.com/zx55/dmcp

**Forward and Backward Information Retention for Accurate Binary Neural Networks**

- 论文：https://arxiv.org/abs/1909.10788

- 代码：https://github.com/htqin/IR-Net

**Towards Efficient Model Compression via Learned Global Ranking**

- 论文：https://arxiv.org/abs/1904.12368
- 代码：https://github.com/cmu-enyac/LeGR

**HRank: Filter Pruning using High-Rank Feature Map**

- 论文：http://arxiv.org/abs/2002.10179
- 代码：https://github.com/lmbxmu/HRank 

**GAN Compression: Efficient Architectures for Interactive Conditional GANs**

- 论文：https://arxiv.org/abs/2003.08936

- 代码：https://github.com/mit-han-lab/gan-compression

**Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression**

- 论文：https://arxiv.org/abs/2003.08935

- 代码：https://github.com/ofsoundof/group_sparsity

<a name="Action-Recognition"></a>

# 视频理解/行为识别

**Intra- and Inter-Action Understanding via Temporal Action Parsing**

- 论文：https://arxiv.org/abs/2005.10229
- 主页和数据集：https://sdolivia.github.io/TAPOS/

**3DV: 3D Dynamic Voxel for Action Recognition in Depth Video**

- 论文：https://arxiv.org/abs/2005.05501
- 代码：https://github.com/3huo/3DV-Action

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- 主页：https://sdolivia.github.io/FineGym/
- 论文：https://arxiv.org/abs/2004.06704

**TEA: Temporal Excitation and Aggregation for Action Recognition**

- 论文：https://arxiv.org/abs/2004.01398

- 代码：https://github.com/Phoenix1327/tea-action-recognition

**X3D: Expanding Architectures for Efficient Video Recognition**

- 论文：https://arxiv.org/abs/2004.04730

- 代码：https://github.com/facebookresearch/SlowFast

**Temporal Pyramid Network for Action Recognition**

- 主页：https://decisionforce.github.io/TPN

- 论文：https://arxiv.org/abs/2004.03548 
- 代码：https://github.com/decisionforce/TPN 

## 基于骨架的动作识别

**Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition**

- 论文：https://arxiv.org/abs/2003.14111
- 代码：https://github.com/kenziyuliu/ms-g3d

<a name="Crowd-Counting"></a>

# 人群计数

<a name="Depth-Estimation"></a>

# 深度估计

**Focus on defocus: bridging the synthetic to real domain gap for depth estimation**

- 论文：https://arxiv.org/abs/2005.09623
- 代码：https://github.com/dvl-tum/defocus-net

**Bi3D: Stereo Depth Estimation via Binary Classifications**

- 论文：https://arxiv.org/abs/2005.07274

- 代码：https://github.com/NVlabs/Bi3D

**AANet: Adaptive Aggregation Network for Efficient Stereo Matching**

- 论文：https://arxiv.org/abs/2004.09548
- 代码：https://github.com/haofeixu/aanet

**Towards Better Generalization: Joint Depth-Pose Learning without PoseNet**

- 论文：https://github.com/B1ueber2y/TrianFlow

- 代码：https://github.com/B1ueber2y/TrianFlow

## 单目深度估计

**On the uncertainty of self-supervised monocular depth estimation**

- 论文：https://arxiv.org/abs/2005.06209
- 代码：https://github.com/mattpoggi/mono-uncertainty

**3D Packing for Self-Supervised Monocular Depth Estimation**

- 论文：https://arxiv.org/abs/1905.02693
- 代码：https://github.com/TRI-ML/packnet-sfm
- Demo视频：https://www.bilibili.com/video/av70562892/

**Domain Decluttering: Simplifying Images to Mitigate Synthetic-Real Domain Shift and Improve Depth Estimation**

- 论文：https://arxiv.org/abs/2002.12114
- 代码：https://github.com/yzhao520/ARC

<a name="6DOF"></a>

# 6D目标姿态估计

**MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion**

- 论文：https://arxiv.org/abs/2004.04336
- 代码：https://github.com/wkentaro/morefusion

**EPOS: Estimating 6D Pose of Objects with Symmetries**

主页：http://cmp.felk.cvut.cz/epos

论文：https://arxiv.org/abs/2004.00605

**G2L-Net: Global to Local Network for Real-time 6D Pose Estimation with Embedding Vector Features**

- 论文：https://arxiv.org/abs/2003.11089

- 代码：https://github.com/DC1991/G2L_Net

<a name="Hand-Pose"></a>

# 手势估计

**HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation**

- 论文：https://arxiv.org/abs/2004.00060

- 主页：http://vision.sice.indiana.edu/projects/hopenet

**Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data**

- 论文：https://arxiv.org/abs/2003.09572

- 代码：https://github.com/CalciferZh/minimal-hand

<a name="Saliency"></a>

# 显著性检测

**JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection**

- 论文：https://arxiv.org/abs/2004.08515

- 代码：https://github.com/kerenfu/JLDCF/

**UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders**

- 主页：http://dpfan.net/d3netbenchmark/

- 论文：https://arxiv.org/abs/2004.05763
- 代码：https://github.com/JingZhang617/UCNet

<a name="Denoising"></a>

# 去噪

**A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising**

- 论文：https://arxiv.org/abs/2003.12751

- 代码：https://github.com/Vandermode/NoiseModel

**CycleISP: Real Image Restoration via Improved Data Synthesis**

- 论文：https://arxiv.org/abs/2003.07761

- 代码：https://github.com/swz30/CycleISP

<a name="Deraining"></a>

# 去雨

**Multi-Scale Progressive Fusion Network for Single Image Deraining**

- 论文：https://arxiv.org/abs/2003.10985

- 代码：https://github.com/kuihua/MSPFN

<a name="Deblurring"></a>

# 去模糊

## 视频去模糊

**Cascaded Deep Video Deblurring Using Temporal Sharpness Prior**

- 主页：https://csbhr.github.io/projects/cdvd-tsp/index.html 
- 论文：https://arxiv.org/abs/2004.02501 
- 代码：https://github.com/csbhr/CDVD-TSP

<a name="Dehazing"></a>

# 去雾

**Multi-Scale Boosted Dehazing Network with Dense Feature Fusion**

- 论文：https://arxiv.org/abs/2004.13388

- 代码：https://github.com/BookerDeWitt/MSBDN-DFF

<a name="Feature"></a>

# 特征点检测与描述

**ASLFeat: Learning Local Features of Accurate Shape and Localization**

- 论文：https://arxiv.org/abs/2003.10071

- 代码：https://github.com/lzx551402/aslfeat

<a name="VQA"></a>

# 视觉问答(VQA)

**VC R-CNN：Visual Commonsense R-CNN** 

- 论文：https://arxiv.org/abs/2002.12204
- 代码：https://github.com/Wangt-CN/VC-R-CNN

<a name="VideoQA"></a>

# 视频问答(VideoQA)

**Hierarchical Conditional Relation Networks for Video Question Answering**

- 论文：https://arxiv.org/abs/2002.10698
- 代码：https://github.com/thaolmk54/hcrn-videoqa

<a name="VLN"></a>

# 视觉语言导航

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training**

- 论文：https://arxiv.org/abs/2002.10638
- 代码（即将开源）：https://github.com/weituo12321/PREVALENT

<a name="Video-Compression"></a>

# 视频压缩

**Learning for Video Compression with Hierarchical Quality and Recurrent Enhancement**

- 论文：https://arxiv.org/abs/2003.01966 
- 代码：https://github.com/RenYang-home/HLVC

<a name="Video-Frame-Interpolation"></a>

# 视频插值

**Space-Time-Aware Multi-Resolution Video Enhancement**

- 主页：https://alterzero.github.io/projects/STAR.html
- 论文：http://arxiv.org/abs/2003.13170
- 代码：https://github.com/alterzero/STARnet

**Scene-Adaptive Video Frame Interpolation via Meta-Learning**

- 论文：https://arxiv.org/abs/2004.00779
- 代码：https://github.com/myungsub/meta-interpolation

**Softmax Splatting for Video Frame Interpolation**

- 主页：http://sniklaus.com/papers/softsplat
- 论文：https://arxiv.org/abs/2003.05534
- 代码：https://github.com/sniklaus/softmax-splatting

<a name="Style-Transfer"></a>

# 风格迁移

**Diversified Arbitrary Style Transfer via Deep Feature Perturbation**

- 论文：https://arxiv.org/abs/1909.08223
- 代码：https://github.com/EndyWon/Deep-Feature-Perturbation

**Collaborative Distillation for Ultra-Resolution Universal Style Transfer**

- 论文：https://arxiv.org/abs/2003.08436

- 代码：https://github.com/mingsun-tse/collaborative-distillation

<a name="Lane-Detection"></a>

# 车道线检测

**Inter-Region Affinity Distillation for Road Marking Segmentation**

- 论文：https://arxiv.org/abs/2004.05304
- 代码：https://github.com/cardwing/Codes-for-IntRA-KD

<a name="HOI"></a>

# "人-物"交互(HOT)检测

**Detailed 2D-3D Joint Representation for Human-Object Interaction**

- 论文：https://arxiv.org/abs/2004.08154

- 代码：https://github.com/DirtyHarryLYL/DJ-RN

**Cascaded Human-Object Interaction Recognition**

- 论文：https://arxiv.org/abs/2003.04262

- 代码：https://github.com/tfzhou/C-HOI

**VSGNet: Spatial Attention Network for Detecting Human Object Interactions Using Graph Convolutions**

- 论文：https://arxiv.org/abs/2003.05541
- 代码：https://github.com/ASMIftekhar/VSGNet

<a name="HTP"></a>

# 行人轨迹预测

**Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction**

- 论文：https://arxiv.org/abs/2002.11927 
- 代码：https://github.com/abduallahmohamed/Social-STGCNN 

<a name="Motion-Predication"></a>

# 运动预测

**Collaborative Motion Prediction via Neural Motion Message Passing**

- 论文：https://arxiv.org/abs/2003.06594
- 代码：https://github.com/PhyllisH/NMMP

**MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird's Eye View Maps**

- 论文：https://arxiv.org/abs/2003.06754

- 代码：https://github.com/pxiangwu/MotionNet

<a name="Virtual-Try-On"></a>

# 虚拟试衣

**Towards Photo-Realistic Virtual Try-On by Adaptively Generating↔Preserving Image Content**

- 论文：https://arxiv.org/abs/2003.05863
- 代码：https://github.com/switchablenorms/DeepFashion_Try_On

<a name="HDR"></a>

# HDR

**Single-Image HDR Reconstruction by Learning to Reverse the Camera Pipeline**

- 主页：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR

- 论文下载链接：https://www.cmlab.csie.ntu.edu.tw/~yulunliu/SingleHDR_/00942.pdf

- 代码：https://github.com/alex04072000/SingleHDR

<a name="AE"></a>

# 对抗样本

**Towards Large yet Imperceptible Adversarial Image Perturbations with Perceptual Color Distance**

- 论文：https://arxiv.org/abs/1911.02466
- 代码：https://github.com/ZhengyuZhao/PerC-Adversarial 

<a name="DC"></a>

# 深度补全

**Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End**

论文：https://arxiv.org/abs/2006.03349

代码：https://github.com/abdo-eldesokey/pncnn

<a name="SSC"></a>

# 语义场景补全

**3D Sketch-aware Semantic Scene Completion via Semi-supervised Structure Prior**

- 论文：https://arxiv.org/abs/2003.14052
- 代码：https://github.com/charlesCXK/3D-SketchAware-SSC 

<a name="Datasets"></a>

# 数据集

**Open Compound Domain Adaptation**

- 主页：https://liuziwei7.github.io/projects/CompoundDomain.html
- 数据集：https://drive.google.com/drive/folders/1_uNTF8RdvhS_sqVTnYx17hEOQpefmE2r?usp=sharing
- 论文：https://arxiv.org/abs/1909.03403
- 代码：https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA

**Intra- and Inter-Action Understanding via Temporal Action Parsing**

- 论文：https://arxiv.org/abs/2005.10229
- 主页和数据集：https://sdolivia.github.io/TAPOS/

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**

- 论文下载链接：https://arxiv.org/abs/2005.09973

- 代码和数据集：https://github.com/Anymake/DRN_CVPR2020

**COCAS: A Large-Scale Clothes Changing Person Dataset for Re-identification**

- 论文：https://arxiv.org/abs/2005.07862

- 数据集：暂无

**KeypointNet: A Large-scale 3D Keypoint Dataset Aggregated from Numerous Human Annotations**

- 论文：https://arxiv.org/abs/2002.12687

- 数据集：https://github.com/qq456cvb/KeypointNet

**MSeg: A Composite Dataset for Multi-domain Semantic Segmentation**

- 论文：http://vladlen.info/papers/MSeg.pdf
- 代码：https://github.com/mseg-dataset/mseg-api

**AvatarMe: Realistically Renderable 3D Facial Reconstruction "in-the-wild"**

- 论文：https://arxiv.org/abs/2003.13845
- 数据集：https://github.com/lattas/AvatarMe

**Learning to Autofocus**

- 论文：https://arxiv.org/abs/2004.12260
- 数据集：暂无

**FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction**

- 论文：https://arxiv.org/abs/2003.13989
- 代码：https://github.com/zhuhao-nju/facescape

**Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data**

- 论文下载链接：https://arxiv.org/abs/2004.01166

- 代码：https://github.com/Healthcare-Robotics/bodies-at-rest
- 数据集：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML

**FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding**

- 主页：https://sdolivia.github.io/FineGym/
- 论文：https://arxiv.org/abs/2004.06704

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- 主页：https://anyirao.com/projects/SceneSeg.html

- 论文下载链接：https://arxiv.org/abs/2004.02678

- 代码：https://github.com/AnyiRao/SceneSeg

**Deep Homography Estimation for Dynamic Scenes**

- 论文：https://arxiv.org/abs/2004.02132

- 数据集：https://github.com/lcmhoang/hmg-dynamics

**Assessing Image Quality Issues for Real-World Problems**

- 主页：https://vizwiz.org/tasks-and-datasets/image-quality-issues/
- 论文：https://arxiv.org/abs/2003.12511

**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**

- 论文：https://arxiv.org/abs/2003.10608
- 代码和数据集：https://github.com/Jyouhou/UnrealText/

**PANDA: A Gigapixel-level Human-centric Video Dataset**

- 论文：https://arxiv.org/abs/2003.04852

- 数据集：http://www.panda-dataset.com/

**IntrA: 3D Intracranial Aneurysm Dataset for Deep Learning**

- 论文：https://arxiv.org/abs/2003.02920
- 数据集：https://github.com/intra3d2019/IntrA

**Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS**

- 论文：https://arxiv.org/abs/2003.03972
- 数据集：暂无

<a name="Others"></a>

# 其他

**Open Compound Domain Adaptation**

- 主页：https://liuziwei7.github.io/projects/CompoundDomain.html
- 数据集：https://drive.google.com/drive/folders/1_uNTF8RdvhS_sqVTnYx17hEOQpefmE2r?usp=sharing
- 论文：https://arxiv.org/abs/1909.03403
- 代码：https://github.com/zhmiao/OpenCompoundDomainAdaptation-OCDA

**Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision**

- 论文：http://www.cvlibs.net/publications/Niemeyer2020CVPR.pdf

- 代码：https://github.com/autonomousvision/differentiable_volumetric_rendering

**QEBA: Query-Efficient Boundary-Based Blackbox Attack**

- 论文：https://arxiv.org/abs/2005.14137
- 代码：https://github.com/AI-secure/QEBA

**Equalization Loss for Long-Tailed Object Recognition**

- 论文：https://arxiv.org/abs/2003.05176
- 代码：https://github.com/tztztztztz/eql.detectron2

**Instance-aware Image Colorization**

- 主页：https://ericsujw.github.io/InstColorization/
- 论文：https://arxiv.org/abs/2005.10825
- 代码：https://github.com/ericsujw/InstColorization

**Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting**

- 论文：https://arxiv.org/abs/2005.09704

- 代码：https://github.com/Atlas200dk/sample-imageinpainting-HiFill

**Where am I looking at? Joint Location and Orientation Estimation by Cross-View Matching**

- 论文：https://arxiv.org/abs/2005.03860
- 代码：https://github.com/shiyujiao/cross_view_localization_DSM

**Epipolar Transformers**

- 论文：https://arxiv.org/abs/2005.04551

- 代码：https://github.com/yihui-he/epipolar-transformers 

**Bringing Old Photos Back to Life**

- 主页：http://raywzy.com/Old_Photo/
- 论文：https://arxiv.org/abs/2004.09484

**MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask**

- 论文：https://arxiv.org/abs/2003.10955 

- 代码：https://github.com/microsoft/MaskFlownet 

**Self-Supervised Viewpoint Learning from Image Collections**

- 论文：https://arxiv.org/abs/2004.01793
- 论文2：https://research.nvidia.com/sites/default/files/pubs/2020-03_Self-Supervised-Viewpoint-Learning/SSV-CVPR2020.pdf 
- 代码：https://github.com/NVlabs/SSV 

**Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations**

- Oral

- 论文：https://arxiv.org/abs/2003.12237 
- 代码：https://github.com/cuishuhao/BNM 

**Towards Learning Structure via Consensus for Face Segmentation and Parsing**

- 论文：https://arxiv.org/abs/1911.00957
- 代码：https://github.com/isi-vista/structure_via_consensus

**Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging**

- Oral
- 论文：https://arxiv.org/abs/2003.13654

- 代码：https://github.com/liuyang12/PnP-SCI

**Lightweight Photometric Stereo for Facial Details Recovery**

- 论文：https://arxiv.org/abs/2003.12307
- 代码：https://github.com/Juyong/FacePSNet

**Footprints and Free Space from a Single Color Image**

- 论文：https://arxiv.org/abs/2004.06376

- 代码：https://github.com/nianticlabs/footprints

**Self-Supervised Monocular Scene Flow Estimation**

- 论文：https://arxiv.org/abs/2004.04143
- 代码：https://github.com/visinf/self-mono-sf

**Quasi-Newton Solver for Robust Non-Rigid Registration**

- 论文：https://arxiv.org/abs/2004.04322
- 代码：https://github.com/Juyong/Fast_RNRR

**A Local-to-Global Approach to Multi-modal Movie Scene Segmentation**

- 主页：https://anyirao.com/projects/SceneSeg.html

- 论文下载链接：https://arxiv.org/abs/2004.02678

- 代码：https://github.com/AnyiRao/SceneSeg

**DeepFLASH: An Efficient Network for Learning-based Medical Image Registration**

- 论文：https://arxiv.org/abs/2004.02097

- 代码：https://github.com/jw4hv/deepflash

**Self-Supervised Scene De-occlusion**

- 主页：https://xiaohangzhan.github.io/projects/deocclusion/
- 论文：https://arxiv.org/abs/2004.02788
- 代码：https://github.com/XiaohangZhan/deocclusion

**Polarized Reflection Removal with Perfect Alignment in the Wild** 

- 主页：https://leichenyang.weebly.com/project-polarized.html
- 代码：https://github.com/ChenyangLEI/CVPR2020-Polarized-Reflection-Removal-with-Perfect-Alignment 

**Background Matting: The World is Your Green Screen**

- 论文：https://arxiv.org/abs/2004.00626
- 代码：http://github.com/senguptaumd/Background-Matting

**What Deep CNNs Benefit from Global Covariance Pooling: An Optimization Perspective**

- 论文：https://arxiv.org/abs/2003.11241

- 代码：https://github.com/ZhangLi-CS/GCP_Optimization

**Look-into-Object: Self-supervised Structure Modeling for Object Recognition**

- 论文：暂无
- 代码：https://github.com/JDAI-CV/LIO 

 **Video Object Grounding using Semantic Roles in Language Description**

- 论文：https://arxiv.org/abs/2003.10606
- 代码：https://github.com/TheShadow29/vognet-pytorch 

**Dynamic Hierarchical Mimicking Towards Consistent Optimization Objectives**

- 论文：https://arxiv.org/abs/2003.10739
- 代码：https://github.com/d-li14/DHM 

**SDFDiff: Differentiable Rendering of Signed Distance Fields for 3D Shape Optimization**

- 论文：http://www.cs.umd.edu/~yuejiang/papers/SDFDiff.pdf
- 代码：https://github.com/YueJiang-nj/CVPR2020-SDFDiff 

**On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location**

- 论文：https://arxiv.org/abs/2003.07064

- 代码：https://github.com/oskyhn/CNNs-Without-Borders

**GhostNet: More Features from Cheap Operations**

- 论文：https://arxiv.org/abs/1911.11907

- 代码：https://github.com/iamhankai/ghostnet

**AdderNet: Do We Really Need Multiplications in Deep Learning?** 

- 论文：https://arxiv.org/abs/1912.13200 
- 代码：https://github.com/huawei-noah/AdderNet

**Deep Image Harmonization via Domain Verification** 

- 论文：https://arxiv.org/abs/1911.13239 
- 代码：https://github.com/bcmi/Image_Harmonization_Datasets

**Blurry Video Frame Interpolation**

- 论文：https://arxiv.org/abs/2002.12259 
- 代码：https://github.com/laomao0/BIN

**Extremely Dense Point Correspondences using a Learned Feature Descriptor**

- 论文：https://arxiv.org/abs/2003.00619 
- 代码：https://github.com/lppllppl920/DenseDescriptorLearning-Pytorch

**Filter Grafting for Deep Neural Networks**

- 论文：https://arxiv.org/abs/2001.05868
- 代码：https://github.com/fxmeng/filter-grafting
- 论文解读：https://www.zhihu.com/question/372070853/answer/1041569335

**Action Segmentation with Joint Self-Supervised Temporal Domain Adaptation**

- 论文：https://arxiv.org/abs/2003.02824 
- 代码：https://github.com/cmhungsteve/SSTDA

**Detecting Attended Visual Targets in Video**

- 论文：https://arxiv.org/abs/2003.02501 

- 代码：https://github.com/ejcgt/attention-target-detection

**Deep Image Spatial Transformation for Person Image Generation**

- 论文：https://arxiv.org/abs/2003.00696 
- 代码：https://github.com/RenYurui/Global-Flow-Local-Attention

 **Rethinking Zero-shot Video Classification: End-to-end Training for Realistic Applications** 

- 论文：https://arxiv.org/abs/2003.01455
- 代码：https://github.com/bbrattoli/ZeroShotVideoClassification

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/Anonymous20192020/Anonymous_CVPR5767

https://github.com/avirambh/ScopeFlow

https://github.com/csbhr/CDVD-TSP

https://github.com/ymcidence/TBH

https://github.com/yaoyao-liu/mnemonics

https://github.com/meder411/Tangent-Images

https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

https://github.com/sjmoran/deep_local_parametric_filters

https://github.com/charlesCXK/3D-SketchAware-SSC

https://github.com/bermanmaxim/AOWS

https://github.com/dc3ea9f/look-into-object 

<a name="Not-Sure"></a>

# 不确定中没中

**FADNet: A Fast and Accurate Network for Disparity Estimation**

- 论文：还没出来
- 代码：https://github.com/HKBU-HPML/FADNet

https://github.com/rFID-submit/RandomFID：不确定中没中

https://github.com/JackSyu/AE-MSR：不确定中没中

https://github.com/fastconvnets/cvpr2020：不确定中没中

https://github.com/aimagelab/meshed-memory-transformer：不确定中没中

https://github.com/TWSFar/CRGNet：不确定中没中

https://github.com/CVPR-2020/CDARTS：不确定中没中

https://github.com/anucvml/ddn-cvprw2020：不确定中没中

https://github.com/dl-model-recommend/model-trust：不确定中没中

https://github.com/apratimbhattacharyya18/CVPR-2020-Corr-Prior：不确定中没中

https://github.com/onetcvpr/O-Net：不确定中没中

https://github.com/502463708/Microcalcification_Detection：不确定中没中

https://github.com/anonymous-for-review/cvpr-2020-deep-smoke-machine：不确定中没中

https://github.com/anonymous-for-review/cvpr-2020-smoke-recognition-dataset：不确定中没中

https://github.com/cvpr-nonrigid/dataset：不确定中没中

https://github.com/theFool32/PPBA：不确定中没中

https://github.com/Realtime-Action-Recognition/Realtime-Action-Recognition