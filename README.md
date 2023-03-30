# CVPR 2023 论文和开源项目合集(Papers with Code)

[CVPR 2023](https://cvpr2023.thecvf.com/) 论文和开源项目合集(papers with code)！

**25.78% = 2360 / 9155**

CVPR2023 decisions are now available on OpenReview! This year, wereceived a record number of **9155** submissions (a 12% increase over CVPR2022), and accepted **2360** papers, for a 25.78% acceptance rate.


> 注1：欢迎各位大佬提交issue，分享CVPR 2023论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2019](CVPR2019-Papers-with-Code.md)
> - [CVPR 2020](CVPR2020-Papers-with-Code.md)
> - [CVPR 2021](CVPR2021-Papers-with-Code.md)
> - [CVPR 2022](CVPR2022-Papers-with-Code.md)

如果你想了解最新最优质的的CV论文、开源项目和学习资料，欢迎扫码加入【CVer学术交流群】！互相学习，一起进步~ 

![](CVer学术交流群.png)

# 【CVPR 2023 论文开源目录】

- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [GAN](#GAN)
- [GNN](#GNN)
- [MLP](#MLP)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Diffusion Models(扩散模型)](#Diffusion)
- [Avatars](#Avatars)
- [ReID(重识别)](#ReID)
- [长尾分布(Long-Tail)](#Long-Tail)
- [Vision Transformer](#Vision-Transformer)
- [视觉和语言(Vision-Language)](#VL)
- [自监督学习(Self-supervised Learning)](#SSL)
- [数据增强(Data Augmentation)](#DA)
- [目标检测(Object Detection)](#Object-Detection)
- [目标跟踪(Visual Tracking)](#VT)
- [语义分割(Semantic Segmentation)](#Semantic-Segmentation)
- [实例分割(Instance Segmentation)](#Instance-Segmentation)
- [全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)
- [医学图像分割(Medical Image Segmentation)](#MIS)
- [视频目标分割(Video Object Segmentation)](#VOS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去模糊(Deblur)](#Deblur)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [医学图像(Medical Image)](#Medical-Image)
- [图像生成(Image Generation)](#Image-Generation)
- [视频生成(Video Generation)](#Video-Generation)
- [视频理解(Video Understanding)](#Video-Understanding)
- [行为检测(Action Detection)](#Action-Detection)
- [文本检测(Text Detection)](#Text-Detection)
- [知识蒸馏(Knowledge Distillation)](#KD)
- [模型剪枝(Model Pruning)](#Pruning)
- [图像压缩(Image Compression)](#IC)
- [异常检测(Anomaly Detection)](#AD)
- [三维重建(3D Reconstruction)](#3D-Reconstruction)
- [深度估计(Depth Estimation)](#Depth-Estimation)
- [轨迹预测(Trajectory Prediction)](#TP)
- [图像描述(Image Captioning)](#Image-Captioning)
- [视觉问答(Visual Question Answering)](#VQA)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [场景图生成(Scene Graph Generation)](#SGG)
- [数据集(Datasets)](#Datasets)
- [新任务(New Tasks)](#New-Tasks)
- [其他(Others)](#Others)

<a name="Backbone"></a>

# Backbone

**Integrally Pre-Trained Transformer Pyramid Networks** 

- Paper: https://arxiv.org/abs/2211.12735
- Code: https://github.com/sunsmarterjie/iTPN

**Stitchable Neural Networks**

- Homepage: https://snnet.github.io/
- Paper: https://arxiv.org/abs/2302.06586
- Code: https://github.com/ziplab/SN-Net

**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**

- Paper: https://arxiv.org/abs/2303.03667
- Code: https://github.com/JierunChen/FasterNet 

**BiFormer: Vision Transformer with Bi-Level Routing Attention**

- Paper: None
- Code: https://github.com/rayleizhu/BiFormer 

**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**

- Paper: https://arxiv.org/abs/2303.02165
- Code: https://github.com/alibaba/lightweight-neural-architecture-search 

**Vision Transformer with Super Token Sampling**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

Hard Patches Mining for Masked Image Modeling

- Paper: None
- Code: None

<a name="CLIP"></a>

# CLIP

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="MAE"></a>

# MAE

**Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders** 

- Paper: https://arxiv.org/abs/2212.06785
- Code: https://github.com/ZrrSkywalker/I2P-MAE

**Generic-to-Specific Distillation of Masked Autoencoders**

- Paper: https://arxiv.org/abs/2302.14771
- Code: https://github.com/pengzhiliang/G2SD

<a name="GAN"></a>

# GAN

**DeltaEdit: Exploring Text-free Training for Text-driven Image Manipulation**

- Paper: https://arxiv.org/abs/2303.06285
- Code: https://github.com/Yueming6568/DeltaEdit 

<a name="NeRF"></a>

# NeRF

**NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior**

- Home: https://nope-nerf.active.vision/
- Paper: https://arxiv.org/abs/2212.07388
- Code: None

**Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures**

- Paper: https://arxiv.org/abs/2211.07600
- Code: https://github.com/eladrich/latent-nerf

**NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via Novel-View Synthesis**

- Paper: https://arxiv.org/abs/2301.08556
- Code: None

**Panoptic Lifting for 3D Scene Understanding with Neural Fields**

- Homepage: https://nihalsid.github.io/panoptic-lifting/
- Paper: https://arxiv.org/abs/2212.09802
- Code: None

**NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer**

- Homepage: https://redrock303.github.io/nerflix/
- Paper: https://arxiv.org/abs/2303.06919 
- Code: None

<a name="DETR"></a>

# DETR

**DETRs with Hybrid Matching**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

<a name="NAS"></a>

# NAS

**PA&DA: Jointly Sampling PAth and DAta for Consistent NAS**

- Paper: https://arxiv.org/abs/2302.14772
- Code: https://github.com/ShunLu91/PA-DA

<a name="Avatars"></a>

# Avatars

**Structured 3D Features for Reconstructing Relightable and Animatable Avatars**

- Homepage: https://enriccorona.github.io/s3f/
- Paper: https://arxiv.org/abs/2212.06820
- Code: None
- Demo: https://www.youtube.com/watch?v=mcZGcQ6L-2s

<a name="ReID"></a>

# ReID(重识别)

**Clothing-Change Feature Augmentation for Person Re-Identification**

- Paper: None
- Code: None

**MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID**

- Paper: https://arxiv.org/abs/2303.07065
- Code: https://github.com/vimar-gu/MSINet

<a name="Diffusion"></a>

# Diffusion Models(扩散模型)

**Video Probabilistic Diffusion Models in Projected Latent Space** 

- Homepage: https://sihyun.me/PVDM/
- Paper: https://arxiv.org/abs/2302.07685
- Code: https://github.com/sihyun-yu/PVDM

**Solving 3D Inverse Problems using Pre-trained 2D Diffusion Models**

- Paper: https://arxiv.org/abs/2211.10655
- Code: None

**Imagic: Text-Based Real Image Editing with Diffusion Models**

- Homepage: https://imagic-editing.github.io/
- Paper: https://arxiv.org/abs/2210.09276
- Code: None

**Parallel Diffusion Models of Operator and Image for Blind Inverse Problems**

- Paper: https://arxiv.org/abs/2211.10656
- Code: None

**DiffRF: Rendering-guided 3D Radiance Field Diffusion**

- Homepage: https://sirwyver.github.io/DiffRF/
- Paper: https://arxiv.org/abs/2212.01206
- Code: None

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising**

- Homepage: https://aminshabani.github.io/housediffusion/
- Paper: https://arxiv.org/abs/2211.13287
- Code: https://github.com/aminshabani/house_diffusion 

**TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**

- Paper: https://arxiv.org/abs/2303.05762
- Code: https://github.com/chenweixin107/TrojDiff

**Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption**

- Paper: https://arxiv.org/abs/2207.03442
- Code: https://github.com/shiyegao/DDA 

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**

- Paper: https://arxiv.org/abs/2303.06885
- Code: None

# Vision Transformer

**Integrally Pre-Trained Transformer Pyramid Networks** 

- Paper: https://arxiv.org/abs/2211.12735
- Code: https://github.com/sunsmarterjie/iTPN

**Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors**

- Homepage: https://niessnerlab.org/projects/hou2023mask3d.html
- Paper: https://arxiv.org/abs/2302.14746
- Code: None

**Learning Trajectory-Aware Transformer for Video Super-Resolution**

- Paper: https://arxiv.org/abs/2204.04216
- Code: https://github.com/researchmm/TTVSR

**Vision Transformers are Parameter-Efficient Audio-Visual Learners**

- Homepage: https://yanbo.ml/project_page/LAVISH/
- Code: https://github.com/GenjiB/LAVISH

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

**BiFormer: Vision Transformer with Bi-Level Routing Attention**

- Paper: https://arxiv.org/abs/2303.08810
- Code: https://github.com/rayleizhu/BiFormer

**Vision Transformer with Super Token Sampling**

- Paper: https://arxiv.org/abs/2211.11167
- Code: https://github.com/hhb072/SViT

**BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision**

- Paper: https://arxiv.org/abs/2211.10439
- Code: None

BAEFormer: Bi-directional and Early Interaction Transformers for Bird’s Eye View Semantic Segmentation

- Paper: None
- Code: None

<a name="VL"></a>

# 视觉和语言(Vision-Language)

**GIVL: Improving Geographical Inclusivity of Vision-Language Models with Pre-Training Methods**

- Paper: https://arxiv.org/abs/2301.01893
- Code: None

**Teaching Structured Vision&Language Concepts to Vision&Language Models**

- Paper: https://arxiv.org/abs/2211.11733
- Code: None

**Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks**

- Paper: https://arxiv.org/abs/2211.09808
- Code: https://github.com/fundamentalvision/Uni-Perceiver

**Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training**

- Paper: https://arxiv.org/abs/2303.00040
- Code: None

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**

- Paper: https://arxiv.org/abs/2303.02483
- Code: None

**Meta-Explore: Exploratory Hierarchical Vision-and-Language Navigation Using Scene Object Spectrum Grounding**

- Homepage: https://rllab-snu.github.io/projects/Meta-Explore/doc.html
- Paper: https://arxiv.org/abs/2303.04077
- Code: None

**All in One: Exploring Unified Video-Language Pre-training**

- Paper: https://arxiv.org/abs/2203.07303
- Code: https://github.com/showlab/all-in-one

**Position-guided Text Prompt for Vision Language Pre-training**

- Paper: https://arxiv.org/abs/2212.09737
- Code: https://github.com/sail-sg/ptp

**EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding**

- Paper: https://arxiv.org/abs/2209.14941
- Code: https://github.com/yanmin-wu/EDA

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**

- Paper: https://arxiv.org/abs/2303.02489
- Code: None

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**

- Paper: https://arxiv.org/abs/2303.02483
- Code: https://github.com/BrandonHanx/FAME-ViL

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection** (Hightlight)

- Paper: https://arxiv.org/abs/2209.01589
- Code: https://github.com/Adamdad/ConsistentTeacher

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**

- Paper: https://arxiv.org/abs/2207.02696
- Code: https://github.com/WongKinYiu/yolov7

**DETRs with Hybrid Matching**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

**Enhanced Training of Query-Based Object Detection via Selective Query Recollection**

- Paper: https://arxiv.org/abs/2212.07593
- Code: https://github.com/Fangyi-Chen/SQR

**Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection**

- Paper: https://arxiv.org/abs/2303.05892
- Code: https://github.com/LutingWang/OADP

<a name="VT"></a>

# 目标跟踪(Object Tracking)

**Simple Cues Lead to a Strong Multi-Object Tracker**

- Paper: https://arxiv.org/abs/2206.04656
- Code: None

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos**

- Paper: https://arxiv.org/abs/2303.07224
- Code: https://github.com/THU-LYJ-Lab/AR-Seg

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

**Label-Free Liver Tumor Segmentation**

- Paper: https://arxiv.org/abs/2210.14845
- Code: https://github.com/MrGiovanni/SyntheticTumors 

<a name="VOS"></a>

# 视频目标分割（Video Object Segmentation）

**Two-shot Video Object Segmentation**

- Paper: https://arxiv.org/abs/2303.12078
- Code: https://github.com/yk-pku/Two-shot-Video-Object-Segmentation

<a name="RIS"></a>

# 参考图像分割(Referring Image Segmentation )

**PolyFormer: Referring Image Segmentation as Sequential Polygon Generation**

- Paper: https://arxiv.org/abs/2302.07387 

- Code: None

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git 

<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)

**DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets**

- Paper: https://arxiv.org/abs/2301.06051
- Code: https://github.com/Haiyang-W/DSVT 

**FrustumFormer: Adaptive Instance-aware Resampling for Multi-view 3D Detection**

- Paper:  https://arxiv.org/abs/2301.04467
- Code: None

**3D Video Object Detection with Learnable Object-Centric Global Optimization**

- Paper: None
- Code: None

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**

- Paper: https://arxiv.org/abs/2303.11203
- Code: https://github.com/l1997i/lim3d 

<a name="3DSSC"></a>

# 3D语义场景补全(3D Semantic Scene Completion)

- Paper: https://arxiv.org/abs/2302.12251
- Code: https://github.com/NVlabs/VoxFormer 

<a name="LLV"></a>

# Low-level Vision

**Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**

- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IR-DIL 

<a name="SR"></a>

# 超分辨率(Video Super-Resolution)

**Super-Resolution Neural Operator**

- Paper: https://arxiv.org/abs/2303.02584
- Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator 

## 视频超分辨率

**Learning Trajectory-Aware Transformer for Video Super-Resolution**

- Paper: https://arxiv.org/abs/2204.04216

- Code: https://github.com/researchmm/TTVSR

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP 

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**

- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**Learning Transferable Spatiotemporal Representations from Natural Script Knowledge**

- Paper: https://arxiv.org/abs/2209.15280
- Code: https://github.com/TencentARC/TVTS

<a name="Action-Detection"></a>

# 行为检测(Action Detection)

**TriDet: Temporal Action Detection with Relative Boundary Modeling**

- Paper: https://arxiv.org/abs/2303.07347
- Code: https://github.com/dingfengshi/TriDet 

<a name="Text-Detection"></a>

# 文本检测(Text Detection)

**DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting**

- Paper: https://arxiv.org/abs/2211.10772
- Code link: https://github.com/ViTAE-Transformer/DeepSolo

<a name="KD"></a>

# 知识蒸馏(Knowledge Distillation)

**Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation**

- Paper: https://arxiv.org/abs/2302.14290
- Code: None

**Generic-to-Specific Distillation of Masked Autoencoders**

- Paper: https://arxiv.org/abs/2302.14771
- Code: https://github.com/pengzhiliang/G2SD

<a name="Pruning"></a>

# 模型剪枝(Model Pruning)

**DepGraph: Towards Any Structural Pruning**

- Paper: https://arxiv.org/abs/2301.12900
- Code: https://github.com/VainF/Torch-Pruning 

<a name="IC"></a>

# 图像压缩(Image Compression)

**Context-Based Trit-Plane Coding for Progressive Image Compression**

- Paper: https://arxiv.org/abs/2303.05715
- Code: https://github.com/seungminjeon-github/CTC

<a name="AD"></a>

# 异常检测(Anomaly Detection)

**Deep Feature In-painting for Unsupervised Anomaly Detection in X-ray Images**

- Paper: https://arxiv.org/abs/2111.13495
- Code: https://github.com/tiangexiang/SQUID 

<a name="3D-Reconstruction"></a>

# 三维重建(3D Reconstruction)

**OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields**

- Paper: https://arxiv.org/abs/2211.12886
- Code: None

**SparsePose: Sparse-View Camera Pose Regression and Refinement**

- Paper: https://arxiv.org/abs/2211.16991
- Code: None

**NeuDA: Neural Deformable Anchor for High-Fidelity Implicit Surface Reconstruction**

- Paper: https://arxiv.org/abs/2303.02375
- Code: None

**Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition**

- Homepage: https://moygcc.github.io/vid2avatar/
- Paper: https://arxiv.org/abs/2302.11566
- Code: https://github.com/MoyGcc/vid2avatar
- Demo: https://youtu.be/EGi47YeIeGQ

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction**

- Paper: https://arxiv.org/abs/2303.05937
- Code: None

**3D Cinemagraphy from a Single Image**

- Homepage: https://xingyi-li.github.io/3d-cinemagraphy/
- Paper: https://arxiv.org/abs/2303.05724
- Code: https://github.com/xingyi-li/3d-cinemagraphy

**Revisiting Rotation Averaging: Uncertainties and Robust Losses**

- Paper: https://arxiv.org/abs/2303.05195
- Code https://github.com/zhangganlin/GlobalSfMpy 

**FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction**

- Paper: https://arxiv.org/abs/2211.13874
- Code: https://github.com/csbhr/FFHQ-UV 

<a name="Depth-Estimation"></a>

# 深度估计(Depth Estimation)

**Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**

- Paper: https://arxiv.org/abs/2211.13202
- Code: https://github.com/noahzn/Lite-Mono 

<a name="TP"></a>

# 轨迹预测(Trajectory Prediction)

**IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction**

- Paper:  https://arxiv.org/abs/2303.00575
- Code: None

<a name="Image-Captioning"></a>

# 图像描述(Image Captioning)

**ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing**

- Paper: https://arxiv.org/abs/2303.02437
- Code: Node

<a name="VQA"></a>

# 视觉问答(Visual Question Answering)

**MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering**

- Paper:  https://arxiv.org/abs/2303.01239
- Code: https://github.com/jingjing12110/MixPHM

<a name="SLR"></a>

# 手语识别(Sign Language Recognition)

**Continuous Sign Language Recognition with Correlation Network**

Paper: https://arxiv.org/abs/2303.03202

Code: https://github.com/hulianyuyy/CorrNet

<a name="Video-Prediction"></a>

# 视频预测(Video Prediction)

**MOSO: Decomposing MOtion, Scene and Object for Video Prediction**

- Paper: https://arxiv.org/abs/2303.03684
- Code: https://github.com/anonymous202203/MOSO

<a name="NVS"></a>

# 新视点合成(Novel View Synthesis)

 **3D Video Loops from Asynchronous Input**

- Homepage: https://limacv.github.io/VideoLoop3D_web/
- Paper: https://arxiv.org/abs/2303.05312
- Code: https://github.com/limacv/VideoLoop3D 

<a name="ZSL"></a>

# Zero-Shot Learning(零样本学习)

**Bi-directional Distribution Alignment for Transductive Zero-Shot Learning**

- Paper: https://arxiv.org/abs/2303.08698
- Code: https://github.com/Zhicaiwww/Bi-VAEGAN

**Semantic Prompt for Few-Shot Learning**

- Paper: None
- Code: None

<a name="Stereo-Matching"></a>

# 立体匹配(Stereo Matching)

**Iterative Geometry Encoding Volume for Stereo Matching**

- Paper: https://arxiv.org/abs/2303.06615
- Code: https://github.com/gangweiX/IGEV

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

**Prototype-based Embedding Network for Scene Graph Generation**

- Paper: https://arxiv.org/abs/2303.07096
- Code: None

<a name="Datasets"></a>

# 数据集(Datasets)

**Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes**

- Paper: https://arxiv.org/abs/2303.02760
- Code: None

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**

- Homepage: https://boheumd.github.io/A2Summ/
- Paper: https://arxiv.org/abs/2303.07284
- Code: https://github.com/boheumd/A2Summ

<a name="Others"></a>

# 其他(Others)

**Interactive Segmentation as Gaussian Process Classification**

- Paper: https://arxiv.org/abs/2302.14578
- Code: None

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**

- Paper: https://arxiv.org/abs/2302.14677
- Code: None

**SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries**

- Homepage: http://bit.ly/splinecam
- Paper: https://arxiv.org/abs/2302.12828
- Code: None

**SCOTCH and SODA: A Transformer Video Shadow Detection Framework**

- Paper: https://arxiv.org/abs/2211.06885
- Code: None

**DeepMapping2: Self-Supervised Large-Scale LiDAR Map Optimization**

- Homepage: https://ai4ce.github.io/DeepMapping2/
- Paper: https://arxiv.org/abs/2212.06331
- None: https://github.com/ai4ce/DeepMapping2

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None

**Token Turing Machines**

- Paper: https://arxiv.org/abs/2211.09119
- Code: None

**Single Image Backdoor Inversion via Robust Smoothed Classifiers**

- Paper: https://arxiv.org/abs/2303.00215
- Code: https://github.com/locuslab/smoothinv

**To fit or not to fit: Model-based Face Reconstruction and Occlusion Segmentation from Weak Supervision**

- Paper: https://arxiv.org/abs/2106.09614
- Code: https://github.com/unibas-gravis/Occlusion-Robust-MoFA

**HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics**

- Homepage: https://dolorousrtur.github.io/hood/
- Paper: https://arxiv.org/abs/2212.07242
- Code: https://github.com/dolorousrtur/hood
- Demo: https://www.youtube.com/watch?v=cBttMDPrUYY

**A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others**

- Paper: https://arxiv.org/abs/2212.04825
- Code: https://github.com/facebookresearch/Whac-A-Mole.git

**RelightableHands: Efficient Neural Relighting of Articulated Hand Models**

- Homepage: https://sh8.io/#/relightable_hands
- Paper: https://arxiv.org/abs/2302.04866
- Code: None
- Demo: https://sh8.io/static/media/teacher_video.923d87957fe0610730c2.mp4

**Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation**

- Paper: https://arxiv.org/abs/2303.00914
- Code: None

**Demystifying Causal Features on Adversarial Examples and Causal Inoculation for Robust Network by Adversarial Instrumental Variable Regression**

- Paper: https://arxiv.org/abs/2303.01052
- Code: None

**UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy**

- Paper: https://arxiv.org/abs/2303.00938
- Code: None

**Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness**

- Paper: https://arxiv.org/abs/2303.00971
- Code: https://github.com/zhijieshen-bjtu/DOPNet

**Learning Neural Parametric Head Models**

- Homepage: https://simongiebenhain.github.io/NPHM)
- Paper: https://arxiv.org/abs/2212.02761
- Code: None

**A Meta-Learning Approach to Predicting Performance and Data Requirements**

- Paper: https://arxiv.org/abs/2303.01598
- Code: None

**MACARONS: Mapping And Coverage Anticipation with RGB Online Self-Supervision**

- Homepage: https://imagine.enpc.fr/~guedona/MACARONS/
- Paper: https://arxiv.org/abs/2303.03315
- Code: None

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**HairStep: Transfer Synthetic to Real Using Strand and Depth Maps for Single-View 3D Hair Modeling**

- Paper: https://arxiv.org/abs/2303.02700
- Code: None

**Decompose, Adjust, Compose: Effective Normalization by Playing with Frequency for Domain Generalization**

- Paper: https://arxiv.org/abs/2303.02328
- Code: None

**Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization**

- Paper: https://arxiv.org/abs/2303.03108
- Code: None

**Unlearnable Clusters: Towards Label-agnostic Unlearnable Examples**

- Paper: https://arxiv.org/abs/2301.01217
- Code: https://github.com/jiamingzhang94/Unlearnable-Clusters 

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**

- Paper: https://arxiv.org/abs/2303.04249
- Code: None

**UniHCP: A Unified Model for Human-Centric Perceptions**

- Paper: https://arxiv.org/abs/2303.02936
- Code: https://github.com/OpenGVLab/UniHCP

**CUDA: Convolution-based Unlearnable Datasets**

- Paper: https://arxiv.org/abs/2303.04278
- Code: https://github.com/vinusankars/Convolution-based-Unlearnability

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**

- Paper: https://arxiv.org/abs/2303.03052
- Code: None

**AdaptiveMix: Robust Feature Representation via Shrinking Feature Space**

- Paper: https://arxiv.org/abs/2303.01559
- Code: https://github.com/WentianZhang-ML/AdaptiveMix 

**Physical-World Optical Adversarial Attacks on 3D Face Recognition**

- Paper: https://arxiv.org/abs/2205.13412
- Code: https://github.com/PolyLiYJ/SLAttack.git

**DPE: Disentanglement of Pose and Expression for General Video Portrait Editing**

- Paper: https://arxiv.org/abs/2301.06281
- Code: https://carlyx.github.io/DPE/ 

**SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation**

- Paper: https://arxiv.org/abs/2211.12194
- Code: https://github.com/Winfredy/SadTalker

**Intrinsic Physical Concepts Discovery with Object-Centric Predictive Models**

- Paper: None
- Code: None

**Sharpness-Aware Gradient Matching for Domain Generalization**

- Paper: None
- Code: https://github.com/Wang-pengfei/SAGM

**Mind the Label-shift for Augmentation-based Graph Out-of-distribution Generalization**

- Paper: None
- Code: None

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**

- Homepage:  https://chenyanglei.github.io/deflicker 
- Paper: None
- Code: None

**RiDDLE: Reversible and Diversified De-identification with Latent Encryptor**

- Paper: None
- Code:  https://github.com/ldz666666/RiDDLE 

**PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation**

- Paper: https://arxiv.org/abs/2303.07337
- Code: None

**Upcycling Models under Domain and Category Shift**

- Paper: https://arxiv.org/abs/2303.07110
- Code: https://github.com/ispc-lab/GLC

**Modality-Agnostic Debiasing for Single Domain Generalization**

- Paper: https://arxiv.org/abs/2303.07123
- Code: None

**Progressive Open Space Expansion for Open-Set Model Attribution**

- Paper: https://arxiv.org/abs/2303.06877
- Code: None

**Dynamic Neural Network for Multi-Task Learning Searching across Diverse Network Topologies**

- Paper: https://arxiv.org/abs/2303.06856
- Code: None

**GFPose: Learning 3D Human Pose Prior with Gradient Fields**

- Paper: https://arxiv.org/abs/2212.08641
- Code: https://github.com/Embracing/GFPose 

**PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment**

- Paper: https://arxiv.org/abs/2303.11526
- Code: https://github.com/Zhang-VISLab

**Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings**

- Paper: https://arxiv.org/abs/2303.11502
- Code: None

**Boundary Unlearning**

- Paper: https://arxiv.org/abs/2303.11570
- Code: None
