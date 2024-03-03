# CVPR 2024 论文和开源项目合集(Papers with Code)

CVPR 2024 decisions are now available on OpenReview！


> 注1：欢迎各位大佬提交issue，分享CVPR 2024论文和开源项目！
>
> 注2：关于往年CV顶会论文以及其他优质CV论文和大盘点，详见： https://github.com/amusi/daily-paper-computer-vision
>
> - [CVPR 2019](CVPR2019-Papers-with-Code.md)
> - [CVPR 2020](CVPR2020-Papers-with-Code.md)
> - [CVPR 2021](CVPR2021-Papers-with-Code.md)
> - [CVPR 2022](CVPR2022-Papers-with-Code.md)
> - [CVPR 2023](CVPR2022-Papers-with-Code.md)

欢迎扫码加入【CVer学术交流群】，这是最大的计算机视觉AI知识星球！每日更新，第一时间分享最新最前沿的计算机视觉、AI绘画、图像处理、深度学习、自动驾驶、医疗影像和AIGC等方向的学习资料，学起来！

![](CVer学术交流群.png)

# 【CVPR 2024 论文开源目录】

- [3DGS(Gaussian Splatting)](#3DGS)
- [Avatars](#Avatars)
- [Backbone](#Backbone)
- [CLIP](#CLIP)
- [MAE](#MAE)
- [Embodied AI](#Embodied-AI)
- [GAN](#GAN)
- [GNN](#GNN)
- [多模态大语言模型(MLLM)](#MLLM)
- [NAS](#NAS)
- [OCR](#OCR)
- [NeRF](#NeRF)
- [DETR](#DETR)
- [Prompt](#Prompt)
- [Diffusion Models(扩散模型)](#Diffusion)
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
- [医学图像(Medical Image)](#MI)
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
- [自动驾驶(Autonomous Driving)](#Autonomous-Driving)
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

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)

**Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**

- Homepage: https://city-super.github.io/scaffold-gs/
- Paper: https://arxiv.org/abs/2312.00109
- Code: https://github.com/city-super/Scaffold-GS

**GPS-Gaussian: Generalizable Pixel-wise 3D Gaussian Splatting for Real-time Human Novel View Synthesis**

- Homepage: https://shunyuanzheng.github.io/GPS-Gaussian 
- Paper: https://arxiv.org/abs/2312.02155
- Code: https://github.com/ShunyuanZheng/GPS-Gaussian

**GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians**

- Paper: https://arxiv.org/abs/2312.02134
- Code: https://github.com/huliangxiao/GaussianAvatar

**GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting**

- Paper: https://arxiv.org/abs/2311.14521
- Code: https://github.com/buaacyw/GaussianEditor 

**Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction**

- Homepage: https://ingra14m.github.io/Deformable-Gaussians/ 
- Paper: https://arxiv.org/abs/2309.13101
- Code: https://github.com/ingra14m/Deformable-3D-Gaussians

**SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes**

- Homepage: https://yihua7.github.io/SC-GS-web/ 
- Paper: https://arxiv.org/abs/2312.14937
- Code: https://github.com/yihua7/SC-GS

<a name="Avatars"></a>

# Avatars

**GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians**

- Paper: https://arxiv.org/abs/2312.02134
- Code: https://github.com/huliangxiao/GaussianAvatar 

<a name="Backbone"></a>

# Backbone

<a name="CLIP"></a>

# CLIP

**Alpha-CLIP: A CLIP Model Focusing on Wherever You Want**

- Paper: https://arxiv.org/abs/2312.03818
- Code: https://github.com/SunzeY/AlphaCLIP 

<a name="MAE"></a>

# MAE

<a name="Embodied-AI"></a>

# Embodied AI

**EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI**

- Homepage: https://tai-wang.github.io/embodiedscan/
- Paper: https://arxiv.org/abs/2312.16170
- Code: https://github.com/OpenRobotLab/EmbodiedScan

**MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception**

- Homepage: https://iranqin.github.io/MP5.github.io/ 
- Paper: https://arxiv.org/abs/2312.07472
- Code: https://github.com/IranQin/MP5

**LEMON: Learning 3D Human-Object Interaction Relation from 2D Images**

- Paper: https://arxiv.org/abs/2312.08963
- Code: https://github.com/yyvhang/lemon_3d 

<a name="GAN"></a>

# GAN

<a name="OCR"></a>

# OCR

**An Empirical Study of Scaling Law for OCR**

- Paper: https://arxiv.org/abs/2401.00028
- Code: https://github.com/large-ocr-model/large-ocr-model.github.io 

<a name="NeRF"></a>

# NeRF

<a name="DETR"></a>

# DETR

**DETRs Beat YOLOs on Real-time Object Detection**

- Paper: https://arxiv.org/abs/2304.08069
- Code: https://github.com/lyuwenyu/RT-DETR

<a name="Prompt"></a>

# Prompt

<a name="MLLM"></a>

# 多模态大语言模型(MLLM)

**mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration**

- Paper: https://arxiv.org/abs/2311.04257
- Code: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2

**Link-Context Learning for Multimodal LLMs**

- Paper: https://arxiv.org/abs/2308.07891
- Code: https://github.com/isekai-portal/Link-Context-Learning/tree/main 

<a name="NAS"></a>

# NAS

<a name="ReID"></a>

# ReID(重识别)

<a name="Diffusion"></a>

# Diffusion Models(扩散模型)

**InstanceDiffusion: Instance-level Control for Image Generation**

- Homepage: https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/

- Paper: https://arxiv.org/abs/2402.03290
- Code: https://github.com/frank-xwang/InstanceDiffusion

**Residual Denoising Diffusion Models**

- Paper: https://arxiv.org/abs/2308.13712
- Code: https://github.com/nachifur/RDDM

**DeepCache: Accelerating Diffusion Models for Free**

- Paper: https://arxiv.org/abs/2312.00858
- Code: https://github.com/horseee/DeepCache 

<a name="Vision-Transformer"></a>

# Vision Transformer

**TransNeXt: Robust Foveal Visual Perception for Vision Transformers**

- Paper: https://arxiv.org/abs/2311.17132
- Code: https://github.com/DaiShiResearch/TransNeXt

<a name="VL"></a>

# 视觉和语言(Vision-Language)

<a name="Object-Detection"></a>

# 目标检测(Object Detection)

**DETRs Beat YOLOs on Real-time Object Detection**

- Paper: https://arxiv.org/abs/2304.08069
- Code: https://github.com/lyuwenyu/RT-DETR

<a name="VT"></a>

# 目标跟踪(Object Tracking)



<a name="Semantic-Segmentation"></a>

# 语义分割(Semantic Segmentation)

**Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation**

- Paper: https://arxiv.org/abs/2312.04265
- Code: https://github.com/w1oves/Rein

**SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation**

- Paper: https://arxiv.org/abs/2311.15537
- Code: https://github.com/xb534/SED 

<a name="MI"></a>

# 医学图像(Medical Image)

**Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology**

- Paper: https://arxiv.org/abs/2402.17228
- Code: https://github.com/DearCaat/RRT-MIL

**VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis**

- Paper: https://arxiv.org/abs/2402.17300
- Code: https://github.com/Luffy03/VoCo

**ChAda-ViT : Channel Adaptive Attention for Joint Representation Learning of Heterogeneous Microscopy Images**

- Paper: https://arxiv.org/abs/2311.15264
- Code: https://github.com/nicoboou/chada_vit 

<a name="MIS"></a>

# 医学图像分割(Medical Image Segmentation)



<a name="Autonomous-Driving"></a>

# 自动驾驶(Autonomous Driving)

**UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**

- Paper: https://arxiv.org/abs/2310.08370
- Code: https://github.com/Nightmare-n/UniPAD 

<a name="3D-Point-Cloud"></a>

# 3D点云(3D-Point-Cloud)



<a name="3DOD"></a>

# 3D目标检测(3D Object Detection)

**PTT: Point-Trajectory Transformer for Efficient Temporal 3D Object Detection**

- Paper: https://arxiv.org/abs/2312.08371
- Code: https://github.com/kuanchihhuang/PTT

**UniMODE: Unified Monocular 3D Object Detection**

- Paper: https://arxiv.org/abs/2402.18573 

<a name="3DOD"></a>

# 3D语义分割(3D Semantic Segmentation)

<a name="Image-Editing"></a>

# 图像编辑(Image Editing)

**Edit One for All: Interactive Batch Image Editing**

- Homepage: https://thaoshibe.github.io/edit-one-for-all 
- Paper: https://arxiv.org/abs/2401.10219
- Code: https://github.com/thaoshibe/edit-one-for-all

<a name="LLV"></a>

# Low-level Vision

**Residual Denoising Diffusion Models**

- Paper: https://arxiv.org/abs/2308.13712

- Code: https://github.com/nachifur/RDDM

<a name="SR"></a>

# 超分辨率(Video Super-Resolution)

<a name="Denoising"></a>

# 去噪(Denoising)

## 图像去噪(Image Denoising)

<a name="Image-Generation"></a>

# 图像生成(Image Generation)

**InstanceDiffusion: Instance-level Control for Image Generation**

- Homepage: https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/

- Paper: https://arxiv.org/abs/2402.03290
- Code: https://github.com/frank-xwang/InstanceDiffusion

**ECLIPSE: A Resource-Efficient Text-to-Image Prior for Image Generations**

- Homepage: https://eclipse-t2i.vercel.app/
- Paper: https://arxiv.org/abs/2312.04655

- Code: https://github.com/eclipse-t2i/eclipse-inference

**Instruct-Imagen: Image Generation with Multi-modal Instruction**

- Paper: https://arxiv.org/abs/2401.01952

**Residual Denoising Diffusion Models**

- Paper: https://arxiv.org/abs/2308.13712
- Code: https://github.com/nachifur/RDDM

**UniGS: Unified Representation for Image Generation and Segmentation**

- Paper: https://arxiv.org/abs/2312.01985

<a name="Video-Generation"></a>

# 视频生成(Video Generation)

**Vlogger: Make Your Dream A Vlog**

- Paper: https://arxiv.org/abs/2401.09414
- Code: https://github.com/Vchitect/Vlogger

**VBench: Comprehensive Benchmark Suite for Video Generative Models**

- Homepage: https://vchitect.github.io/VBench-project/ 

- Paper: https://arxiv.org/abs/2311.17982
- Code: https://github.com/Vchitect/VBench

<a name="Video-Understanding"></a>

# 视频理解(Video Understanding)

**MVBench: A Comprehensive Multi-modal Video Understanding Benchmark**

Paper: https://arxiv.org/abs/2311.17005

Code: https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2 

# 其他(Others)

**Object Recognition as Next Token Prediction**

- Paper: https://arxiv.org/abs/2312.02142
- Code: https://github.com/kaiyuyue/nxtp

**ParameterNet: Parameters Are All You Need for Large-scale Visual Pretraining of Mobile Networks**

- Paper: https://arxiv.org/abs/2306.14525
- Code: https://parameternet.github.io/ 

**Seamless Human Motion Composition with Blended Positional Encodings**

- Paper: https://arxiv.org/abs/2402.15509
- Code: https://github.com/BarqueroGerman/FlowMDM 

**LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning**

- Homepage:  https://ll3da.github.io/ 

- Paper: https://arxiv.org/abs/2311.18651
- Code: https://github.com/Open3DA/LL3DA

 **CLOVA: A Closed-LOop Visual Assistant with Tool Usage and Update**

- Homepage: https://clova-tool.github.io/ 
- Paper: https://arxiv.org/abs/2312.10908

**MoMask: Generative Masked Modeling of 3D Human Motions**

- Paper: https://arxiv.org/abs/2312.00063
- Code: https://github.com/EricGuo5513/momask-codes 