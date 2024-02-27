# CVPR 2023 论文和开源项目合集(Papers with Code)

[CVPR 2023](https://openaccess.thecvf.com/CVPR2023?day=all) 论文和开源项目合集(papers with code)！

**25.78% = 2360 / 9155**

CVPR 2023 decisions are now available on OpenReview! This year, wereceived a record number of **9155** submissions (a 12% increase over CVPR 2022), and accepted **2360** papers, for a 25.78% acceptance rate.


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
- [Prompt](#Prompt)
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
- [视频实例分割(Video Instance Segmentation)](#VIS)
- [参考图像分割(Referring Image Segmentation)](#RIS)
- [图像抠图(Image Matting)](#Matting)
- [图像编辑(Image Editing)](#Image-Editing)
- [Low-level Vision](#LLV)
- [超分辨率(Super-Resolution)](#SR)
- [去噪(Denoising)](#Denoising)
- [去模糊(Deblur)](#Deblur)
- [3D点云(3D Point Cloud)](#3D-Point-Cloud)
- [3D目标检测(3D Object Detection)](#3DOD)
- [3D语义分割(3D Semantic Segmentation)](#3DSS)
- [3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking)
- [3D语义场景补全(3D Semantic Scene Completion)](#3DSSC)
- [3D配准(3D Registration)](#3D-Registration)
- [3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation)
- [3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation)
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
- [车道线检测(Lane Detection)](#Lane-Detection)
- [图像描述(Image Captioning)](#Image-Captioning)
- [视觉问答(Visual Question Answering)](#VQA)
- [手语识别(Sign Language Recognition)](#SLR)
- [视频预测(Video Prediction)](#Video-Prediction)
- [新视点合成(Novel View Synthesis)](#NVS)
- [Zero-Shot Learning(零样本学习)](#ZSL)
- [立体匹配(Stereo Matching)](#Stereo-Matching)
- [特征匹配(Feature Matching)](#Feature-Matching)
- [场景图生成(Scene Graph Generation)](#SGG)
- [隐式神经表示(Implicit Neural Representations)](#INR)
- [图像质量评价(Image Quality Assessment)](#IQA)
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

**Hard Patches Mining for Masked Image Modeling**

- Paper: None
- Code: None

**SMPConv: Self-moving Point Representations for Continuous Convolution**

- Paper: https://arxiv.org/abs/2304.02330
- Code: https://github.com/sangnekim/SMPConv

**Making Vision Transformers Efficient from A Token Sparsification View**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

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

**HNeRV: A Hybrid Neural Representation for Videos**

- Homepage: https://haochen-rye.github.io/HNeRV
- Paper: https://arxiv.org/abs/2304.02633
- Code: https://github.com/haochen-rye/HNeRV

<a name="DETR"></a>

# DETR

**DETRs with Hybrid Matching**

- Paper: https://arxiv.org/abs/2207.13080
- Code: https://github.com/HDETR

<a name="Prompt"></a>

# Prompt

**Diversity-Aware Meta Visual Prompting**

- Paper: https://arxiv.org/abs/2303.08138
- Code: https://github.com/shikiw/DAM-VP 

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

**Learning Personalized High Quality Volumetric Head Avatars from Monocular RGB Videos**

- Homepage: https://augmentedperception.github.io/monoavatar/
- Paper: https://arxiv.org/abs/2304.01436

<a name="ReID"></a>

# ReID(重识别)

**Clothing-Change Feature Augmentation for Person Re-Identification**

- Paper: None
- Code: None

**MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID**

- Paper: https://arxiv.org/abs/2303.07065
- Code: https://github.com/vimar-gu/MSINet

**Shape-Erased Feature Learning for Visible-Infrared Person Re-Identification**

- Paper: https://arxiv.org/abs/2304.04205
- Code: None

**Large-scale Training Data Search for Object Re-identification**

- Paper: https://arxiv.org/abs/2303.16186
- Code: https://github.com/yorkeyao/SnP 

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

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Generative Diffusion Prior for Unified Image Restoration and Enhancement**

- Paper: https://arxiv.org/abs/2304.01247
- Code: None

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Long-Tail"></a>

# 长尾分布(Long-Tail)

**Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation**

- Paper: https://arxiv.org/abs/2304.01279
- Code: None

<a name="Vision-Transformer"></a>

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

**BAEFormer: Bi-directional and Early Interaction Transformers for Bird’s Eye View Semantic Segmentation**

- Paper: None
- Code: None

**Visual Dependency Transformers: Dependency Tree Emerges from Reversed Attention**

- Paper: https://arxiv.org/abs/2304.03282
- Code: None

**Making Vision Transformers Efficient from A Token Sparsification View**

- Paper: https://arxiv.org/abs/2303.08685
- Code: https://github.com/changsn/STViT-R 

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

**Multi-Modal Representation Learning with Text-Driven Soft Masks**

- Paper: https://arxiv.org/abs/2304.00719
- Code: None

**Learning to Name Classes for Vision and Language Models**

- Paper: https://arxiv.org/abs/2304.01830
- Code: None

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

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

**Joint Visual Grounding and Tracking with Natural Language Specification**

- Paper: https://arxiv.org/abs/2303.12027
- Code: https://github.com/lizhou-cs/JointNLT 

<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos**

- Paper: https://arxiv.org/abs/2303.07224
- Code: https://github.com/THU-LYJ-Lab/AR-Seg

**FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding**

- Paper: https://arxiv.org/abs/2304.02135
- Code: https://github.com/uark-cviu/FREDOM

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)

**Label-Free Liver Tumor Segmentation**

- Paper: https://arxiv.org/abs/2303.14869
- Code: https://github.com/MrGiovanni/SyntheticTumors

**Directional Connectivity-based Segmentation of Medical Images**

- Paper: https://arxiv.org/abs/2304.00145
- Code: https://github.com/Zyun-Y/DconnNet

**Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation**

- Paper: https://arxiv.org/abs/2305.00673
- Code: https://github.com/DeepMed-Lab-ECNU/BCP

**Devil is in the Queries: Advancing Mask Transformers for Real-world Medical Image Segmentation and Out-of-Distribution Localization**

- Paper: https://arxiv.org/abs/2304.00212
- Code: None

**Fair Federated Medical Image Segmentation via Client Contribution Estimation**

- Paper: https://arxiv.org/abs/2303.16520
- Code: https://github.com/NVIDIA/NVFlare/tree/dev/research/fed-ce

**Ambiguous Medical Image Segmentation using Diffusion Models**

- Homepage: https://aimansnigdha.github.io/cimd/
- Paper: https://arxiv.org/abs/2304.04745
- Code: https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models

**Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation**

- Paper: https://arxiv.org/abs/2303.13090
- Code: https://github.com/HengCai-NJU/DeSCO

**MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery**

- Paper: https://arxiv.org/abs/2301.01767
- Code: https://github.com/DeepMed-Lab-ECNU/MagicNet

**MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MCF_Mutual_Correction_Framework_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/WYC-321/MCF

**Rethinking Few-Shot Medical Segmentation: A Vector Quantization View**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Rethinking_Few-Shot_Medical_Segmentation_A_Vector_Quantization_View_CVPR_2023_paper.html
- Code: None

**Pseudo-label Guided Contrastive Learning for Semi-supervised Medical Image Segmentation**

- Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Basak_Pseudo-Label_Guided_Contrastive_Learning_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html
- Code: https://github.com/hritam-98/PatchCL-MedSeg

**SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation**

- Paper: https://arxiv.org/abs/2305.11012
- Code: None

**DoNet: Deep De-overlapping Network for Cytology Instance Segmentation**

- Paper: https://arxiv.org/abs/2303.14373
- Code: https://github.com/DeepDoNet/DoNet

<a name="VOS"></a>

# 视频目标分割（Video Object Segmentation）

**Two-shot Video Object Segmentation**

- Paper: https://arxiv.org/abs/2303.12078
- Code: https://github.com/yk-pku/Two-shot-Video-Object-Segmentation

 **Under Video Object Segmentation Section**

- Paper: https://arxiv.org/abs/2303.07815
- Code: None

<a name="VIS"></a>

# 视频实例分割(Video Instance Segmentation)

**Mask-Free Video Instance Segmentation**

- Paper: https://arxiv.org/abs/2303.15904
- Code: https://github.com/SysCV/MaskFreeVis 

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

**IterativePFN: True Iterative Point Cloud Filtering**

- Paper: https://arxiv.org/abs/2304.01529
- Code: https://github.com/ddsediri/IterativePFN

**Attention-based Point Cloud Edge Sampling**

- Homepage: https://junweizheng93.github.io/publications/APES/APES.html 
- Paper: https://arxiv.org/abs/2302.14673
- Code: https://github.com/JunweiZheng93/APES

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

**Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection**

- Paper: https://arxiv.org/abs/2304.01464
- Code: https://github.com/azhuantou/HSSDA

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**

- Paper: https://arxiv.org/abs/2303.11203
- Code: https://github.com/l1997i/lim3d 

<a name="3DSSC"></a>

# 3D语义场景补全(3D Semantic Scene Completion)

- Paper: https://arxiv.org/abs/2302.12251
- Code: https://github.com/NVlabs/VoxFormer 

<a name="3D-Registration"></a>

# 3D配准(3D Registration)

**Robust Outlier Rejection for 3D Registration with Variational Bayes**

- Paper: https://arxiv.org/abs/2304.01514
- Code: https://github.com/Jiang-HB/VBReg

<a name="3D-Human-Pose-Estimation"></a>

# 3D人体姿态估计(3D Human Pose Estimation)

<a name="3D-Human-Mesh-Estimation"></a>

# 3D人体Mesh估计(3D Human Mesh Estimation)

**3D Human Mesh Estimation from Virtual Markers**

- Paper: https://arxiv.org/abs/2303.11726
- Code: https://github.com/ShirleyMaxx/VirtualMarker 

<a name="LLV"></a>

# Low-level Vision

**Causal-IR: Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**

- Paper: https://arxiv.org/abs/2303.06859
- Code: https://github.com/lixinustc/Casual-IR-DIL 

**Burstormer: Burst Image Restoration and Enhancement Transformer**

- Paper: https://arxiv.org/abs/2304.01194
- Code: http://github.com/akshaydudhane16/Burstormer

<a name="SR"></a>

# 超分辨率(Video Super-Resolution)

**Super-Resolution Neural Operator**

- Paper: https://arxiv.org/abs/2303.02584
- Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator 

## 视频超分辨率

**Learning Trajectory-Aware Transformer for Video Super-Resolution**

- Paper: https://arxiv.org/abs/2204.04216

- Code: https://github.com/researchmm/TTVSR

Denoising<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

**Masked Image Training for Generalizable Deep Image Denoising**

- Paper- : https://arxiv.org/abs/2303.13132
- Code: https://github.com/haoyuc/MaskedDenoising 

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis**

- Paper: https://arxiv.org/abs/2301.12959
- Code: https://github.com/tobran/GALIP 

**MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**

- Paper: https://arxiv.org/abs/2211.09117
- Code: https://github.com/LTH14/mage

**Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation**

- Paper: https://arxiv.org/abs/2304.01816
- Code: None

**Few-shot Semantic Image Synthesis with Class Affinity Transfer**

- Paper: https://arxiv.org/abs/2304.02321
- Code: None

**TopNet: Transformer-based Object Placement Network for Image Compositing**

- Paper: https://arxiv.org/abs/2304.03372
- Code: None

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation**

- Paper: https://arxiv.org/abs/2212.09478
- Code: https://github.com/researchmm/MM-Diffusion

**Conditional Image-to-Video Generation with Latent Flow Diffusion Models**

- Paper: https://arxiv.org/abs/2303.13744
- Code: https://github.com/nihaomiao/CVPR23_LFDM 

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**Learning Transferable Spatiotemporal Representations from Natural Script Knowledge**

- Paper: https://arxiv.org/abs/2209.15280
- Code: https://github.com/TencentARC/TVTS

**Frame Flexible Network**

- Paper: https://arxiv.org/abs/2303.14817
- Code: https://github.com/BeSpontaneous/FFN

**Masked Motion Encoding for Self-Supervised Video Representation Learning**

- Paper: https://arxiv.org/abs/2210.06096
- Code: https://github.com/XinyuSun/MME

**MARLIN: Masked Autoencoder for facial video Representation LearnING**

- Paper: https://arxiv.org/abs/2211.06627
- Code: https://github.com/ControlNet/MARLIN 

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

**A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images**

- Homepage: https://younglbw.github.io/HRN-homepage/ 

- Paper: https://arxiv.org/abs/2302.14434
- Code: https://github.com/youngLBW/HRN

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

**EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning**

- Paper: https://arxiv.org/abs/2303.10876
- Code: https://github.com/MediaBrain-SJTU/EqMotion 

<a name="Lane-Detection"></a>

# 车道线检测(Lane Detection)

**Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection**

- Paper: https://arxiv.org/abs/2301.02371
- Code: https://github.com/tusen-ai/Anchor3DLane

**BEV-LaneDet: An Efficient 3D Lane Detection Based on Virtual Camera via Key-Points**

- Paper:  https://arxiv.org/abs/2210.06006v3 
- Code:  https://github.com/gigo-team/bev_lane_det 

<a name="Image-Captioning"></a>

# 图像描述(Image Captioning)

**ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing**

- Paper: https://arxiv.org/abs/2303.02437
- Code: Node

**Cross-Domain Image Captioning with Discriminative Finetuning**

- Paper: https://arxiv.org/abs/2304.01662
- Code: None

**Model-Agnostic Gender Debiased Image Captioning**

- Paper: https://arxiv.org/abs/2304.03693
- Code: None

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

**Learning the Distribution of Errors in Stereo Matching for Joint Disparity and Uncertainty Estimation**

- Paper: https://arxiv.org/abs/2304.00152
- Code: None

<a name="Feature-Matching"></a>

# 特征匹配(Feature Matching)

**Adaptive Spot-Guided Transformer for Consistent Local Feature Matching**

- Homepage: [https://astr2023.github.io](https://astr2023.github.io/) 
- Paper: https://arxiv.org/abs/2303.16624
- Code: https://github.com/ASTR2023/ASTR

<a name="SGG"></a>

# 场景图生成(Scene Graph Generation)

**Prototype-based Embedding Network for Scene Graph Generation**

- Paper: https://arxiv.org/abs/2303.07096
- Code: None

<a name="INR"></a>

# 隐式神经表示(Implicit Neural Representations)

**Polynomial Implicit Neural Representations For Large Diverse Datasets**

- Paper: https://arxiv.org/abs/2303.11424
- Code: https://github.com/Rajhans0/Poly_INR

<a name="IQA"></a>

# 图像质量评价(Image Quality Assessment)

**Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild**

- Paper: https://arxiv.org/abs/2304.00451
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

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**

- Homepage: https://tarun005.github.io/GeoNet/
- Paper: https://arxiv.org/abs/2303.15443

**CelebV-Text: A Large-Scale Facial Text-Video Dataset**

- Homepage: https://celebv-text.github.io/
- Paper: https://arxiv.org/abs/2303.14717

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

**ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing**

- Paper: https://arxiv.org/abs/2303.17096
- Code: https://github.com/alibaba/easyrobust

**Zero-shot Model Diagnosis**

- Paper: https://arxiv.org/abs/2303.15441
- Code: None

**GeoNet: Benchmarking Unsupervised Adaptation across Geographies**

- Homepage: https://tarun005.github.io/GeoNet/
- Paper: https://arxiv.org/abs/2303.15443

**Quantum Multi-Model Fitting**

- Paper: https://arxiv.org/abs/2303.15444
- Code: https://github.com/FarinaMatteo/qmmf

**DivClust: Controlling Diversity in Deep Clustering**

- Paper: https://arxiv.org/abs/2304.01042
- Code: None

**Neural Volumetric Memory for Visual Locomotion Control**

- Homepage: https://rchalyang.github.io/NVM
- Paper: https://arxiv.org/abs/2304.01201
- Code: https://rchalyang.github.io/NVM

**MonoHuman: Animatable Human Neural Field from Monocular Video**

- Homepage: https://yzmblog.github.io/projects/MonoHuman/
- Paper: https://arxiv.org/abs/2304.02001
- Code: https://github.com/Yzmblog/MonoHuman

**Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion**

- Homepage: https://nv-tlabs.github.io/trace-pace/
- Paper: https://arxiv.org/abs/2304.01893
- Code: None

**Bridging the Gap between Model Explanations in Partially Annotated Multi-label Classification**

- Paper: https://arxiv.org/abs/2304.01804
- Code: None

**HyperCUT: Video Sequence from a Single Blurry Image using Unsupervised Ordering**

- Paper: https://arxiv.org/abs/2304.01686
- Code: None

**On the Stability-Plasticity Dilemma of Class-Incremental Learning**

- Paper: https://arxiv.org/abs/2304.01663
- Code: None

**Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning**

- Paper: https://arxiv.org/abs/2304.01482
- Code: None

**VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution**

- Paper: https://arxiv.org/abs/2304.01434
- Code: https://github.com/jaeill/CVPR23-VNE

**Detecting and Grounding Multi-Modal Media Manipulation**

- Homepage: https://rshaojimmy.github.io/Projects/MultiModal-DeepFake
- Paper: https://arxiv.org/abs/2304.02556
- Code: https://github.com/rshaojimmy/MultiModal-DeepFake

**Meta-causal Learning for Single Domain Generalization**

- Paper: https://arxiv.org/abs/2304.03709
- Code: None

**Disentangling Writer and Character Styles for Handwriting Generation**

- Paper: https://arxiv.org/abs/2303.14736
- Code: https://github.com/dailenson/SDT

**DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects**

- Homepage: https://www.chenbao.tech/dexart/

- Code: https://github.com/Kami-code/dexart-release

**Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision**

- Homepage: https://toytiny.github.io/publication/23-cmflow-cvpr/index.html 
- Paper: https://arxiv.org/abs/2303.00462
- Code: https://github.com/Toytiny/CMFlow

**Marching-Primitives: Shape Abstraction from Signed Distance Function**

- Paper: https://arxiv.org/abs/2303.13190
- Code: https://github.com/ChirikjianLab/Marching-Primitives

**Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision**

- Paper: https://arxiv.org/abs/2303.00885
- Code: None