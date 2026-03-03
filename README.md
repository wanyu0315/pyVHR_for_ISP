# pyVHR_for_ISP
前端图像信号处理对rppg质量影响的实验源码，包含了自己组装的ISP管道、pyVHR框架处理分析等模块
# 🚀 ISP-Pipeline for rPPG Research
ISPpipeline是使用python实现的可配置、白盒化的ISP管道流程，能够完全仿真常见消费、工业级相机的ISP流程，输入为裸raw格式的数据。
1. 项目定位 (Project Positioning)
这是一个专为 微弱生理信号提取（如 rPPG/VHR） 以及 底层计算机视觉科研 设计的模块化图像信号处理（ISP）流水线。

传统的消费级相机和标准的 MP4/H.264 视频压缩算法（如时域预测、色度下采样）会破坏 rPPG 依赖的微弱血管容积脉搏波信号，并且ISP流水黑盒不可见。本项目旨在解决这一痛点，其核心定位是：

提供绝对的控制变量环境：允许研究人员开启、关闭或精细调节从 RAW 数据到 RGB 视频的每一个处理环节（如去马赛克、白平衡、降噪等），是进行 ISP 消融实验（Ablation Study）的完美基准（Baseline）工具。

确保数据物理保真度：采用 FFV1 绝对无损编码和 bgr0 格式，彻底消除视频压缩伪影对生理信号的二次污染，将 rPPG 的性能差异严格限制在 ISP 算法层面。

2. 核心特性 (Core Features)
全链路模块化 ISP：涵盖黑电平校正 (BLC)、坏点校正 (DPC)、白平衡 (AWB)、去马赛克 (Demosaic)、色彩校正矩阵 (CCM)、Gamma 校正、色彩空间转换及 YUV 域增强。

科研级无损视频合成：集成 FFmpeg，默认使用 ffv1 编码器、大上下文模型及 GOP=1（全 I 帧），实现 100% 空间与时域无损。

自动化批量处理：支持遍历根目录下的多个 RAW 视频序列子文件夹，自动跳过已处理帧，支持断点续传。

坏帧动态检测与剔除：内置基于行均值的黑屏/坏帧检测机制，自动使用上一有效帧进行补偿。

实验参数自动留档：每次处理后，自动将当前所有的 ISP 参数及 FFmpeg 编码配置保存为同名 .json 文件，确保科研数据的可追溯性。

3. 环境依赖 (Prerequisites)
Python: 3.7+

系统组件: 必须安装 FFmpeg 并将其添加至系统环境变量 (PATH)。

Python 依赖包:

Bash
pip install numpy opencv-python imageio tqdm rawpy colour-demosaicing tifffile
4. 目录结构说明 (Directory Structure)
在运行代码前，请确保原始数据按以下结构组织：

Plaintext
├── Data_preprocessing/
│   └── defect_report/bad_points_report_longtimevideo/
│       └── defect_map.npy           # 必须：传感器坏点坐标图
├── ISPpipline/
│   └── raw_data/
│       └── baseenvironment_rawframe/# ROOT_INPUT_DIR (根输入目录)
│           ├── video_seq_01/        # 视频序列1
│           │   ├── frame_0001.raw
│           │   └── ...
│           └── video_seq_02/        # 视频序列2
│               ├── frame_0001.raw
│               └── ...
├── Data_for_pyVHR/                  # 视频和JSON日志输出目录
└── main_batch_raw.py                # 主执行脚本
5. 使用方法 (Quick Start)
步骤 1：修改硬件元数据
打开 main_batch_raw.py，根据相机传感器规格修改常量：
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 800
    IMAGE_DTYPE = np.uint16  # 16-bit或8-bit RAW
    BAYER_PATTERN = 'GBRG'   # 传感器的Bayer排列格式
步骤 2：检查路径配置
确认根目录路径（ROOT_INPUT_DIR, ROOT_OUTPUT_FRAME_DIR, ROOT_OUTPUT_VIDEO_DIR）指向本地实际路径。

步骤 3：调整 ISP 参数 (按需)
在 processing_params 字典（约第 93 行开始）中，可以自由开启或关闭各项算法。例如：

测试不同去马赛克算法：修改 'demosaic': {'algorithm': 'AHD'} 或 'bilinear'。

纯线性输出 (无 Gamma 扭曲)：修改 'gammacorrection': {'gamma': 1.0}。

步骤 4：运行脚本
    python main_batch_raw.py
程序将自动扫描所有子文件夹，渲染 PNG 序列，并最终在输出目录生成 .mkv 视频和 .json 参数日志。

6. 路径配置
    ROOT_INPUT_DIR：存放所有 RAW 序列文件夹的根目录 (例如下面有 raw_lzz, raw_test1 等) 
    例如：ROOT_INPUT_DIR = 'ISPpipline/raw_data/baseenvironment_rawframe'

    ROOT_OUTPUT_FRAME_DIR：存放所有处理后 PNG 帧的根目录
    例如：ROOT_OUTPUT_FRAME_DIR = '/home/lizize/pyVHR_for_ISP/ISPpipline/isp_output_frame/sensitivity_analysis/baseenv/demosaic/malvar2004'

    ROOT_OUTPUT_VIDEO_DIR：存放最终输出视频和 JSON 参数文件的根目录
    例如：ROOT_OUTPUT_VIDEO_DIR = 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/demosaic/malvar2004'

    defect_map：坏点map地址
    例如：defect_map = np.load('Data_preprocessing/defect_report/bad_points_report_longtimevideo/defect_map.npy')

7. 参数配置详解 (Configuration Guide)
processing_params 是整个流水线的“大脑”，关键模块配置说明如下：

rawdenoise (RAW域降噪): 默认设为 'algorithm': 'None' 以保留原始高频信息。如需开启，支持配置 gaussian 等算法级联。

whitebalance (白平衡): 默认使用 gray_world_green（以绿光为基准的灰度世界算法），最适合还原人脸肤色。

demosaic (去马赛克):

bilinear: 基础双线性插值（速度快，但高频边缘会模糊）。

AHD: 高级自适应均匀性定向插值（推荐，有效抑制伪彩，保留色彩边缘）。

gammacorrection: 2.2 模拟人眼视觉及常规显示器输出；设为 1.0 则保持物理线性，极力推荐用于 rPPG 严谨分析。