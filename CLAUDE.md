# CLAUDE.md — ISP for pyVHR 实验项目

## 研究背景

本项目研究**前端 ISP 流水线对 rPPG 信号质量的影响**。通过构建白盒、模块化的 Python ISP 流水线，配合 pyVHR 框架和自制探针系统，进行量化消融实验。

## 项目结构

```
pyVHR_for_ISP/
├── ISPpipline/             # [Git子模块] 白盒 ISP 流水线
│   ├── Pipeline/           # ISP 各模块实现
│   ├── ISP_main.py         # 主批量处理入口
│   └── pipeline_probe.py   # ISP 探针模块
├── pyVHR_run_on_video/     # pyVHR 标准分析流程
│   └── analyze_with_pyvhr.py
├── probe_analysis/         # 探针数据分析工具
│   └── probe_ana_cmp.py
├── Data_preprocessing/     # 数据预处理工具
└── Data_for_pyVHR/         # 数据存储
```

## 路径约定

- ISP 输出视频：`Data_for_pyVHR/isp_output_Video/<实验组>/`
- GT 数据：`Data_for_pyVHR/gt_data/gt_<受试者名>/bpms_times_GT.npz`
- 探针数据：`probes_debug/<实验组>/<受试者名>/<探针名>/`
- 分析结果：`rPPGanalyze_res_plots/<实验类型>/<算法>/`
- 探针分析结果：`probe_analysis/probe_analysis_result/<实验组>/`

## 环境依赖

```
# ISP 流水线
numpy, opencv-python, imageio, tqdm, rawpy, colour-demosaicing, tifffile

# rPPG 分析
pyVHR, mediapipe, scipy, matplotlib

# GPU 加速（可选）
cupy, numba, torch

# 系统依赖
FFmpeg（必须在 PATH 中，用于 FFV1 无损视频合成）
```

## 详细规则

@.claude/rules/rppg-analysis.md
@.claude/rules/isp-pipeline.md
@.claude/rules/probe-system.md
@.claude/rules/ablation-study.md
@.claude/rules/code-style.md