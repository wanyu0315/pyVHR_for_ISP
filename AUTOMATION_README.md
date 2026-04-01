# ISP-pyVHR 自动化集成系统使用说明

## ⚠️ 重要：目录结构和命名规范

### 核心映射规则

自动化系统通过**目录名称**进行映射，必须确保以下对应关系：

```
配置文件中的 subject = RAW 目录名
GT 目录名 = gt_{subject}
```

#### 命名规范

1. **RAW 视频帧目录**：`raw_data/baseenv_rawframe/{任意名称}/`
2. **GT 数据目录**：`Data_for_pyVHR/gt_data/gt_{任意名称}/`
3. **配置文件**：`subjects: ["{任意名称}"]`

**关键**：三者中的 `{任意名称}` 必须完全一致。

---

### 示例 1：简单命名（推荐）

```
目录结构：
raw_data/baseenv_rawframe/hyl/
Data_for_pyVHR/gt_data/gt_hyl/

配置文件：
subjects: ["hyl"]
```

### 示例 2：带前缀命名

```
目录结构：
raw_data/baseenv_rawframe/raw_hyl/
Data_for_pyVHR/gt_data/gt_raw_hyl/

配置文件：
subjects: ["raw_hyl"]
```

### 示例 3：带编号命名

```
目录结构：
raw_data/baseenv_rawframe/subject_001/
Data_for_pyVHR/gt_data/gt_subject_001/

配置文件：
subjects: ["subject_001"]
```

---

### 映射流程

```
配置：subjects = ["hyl"]
    ↓
RAW 目录：hyl/
    ↓ ISP 处理
输出视频：hyl_output_8bit.mkv
    ↓ 自动匹配
GT 数据：gt_hyl/bpms_times_GT.npz
```

---

## 完整目录结构示例

```
pyVHR_for_ISP/
├── ISPpipline/
│   └── raw_data/
│       └── baseenv_rawframe/
│           ├── hyl/                    # 受试者 hyl 的 RAW 帧
│           │   ├── frame_0000.raw
│           │   └── ...
│           ├── lj/                     # 受试者 lj 的 RAW 帧
│           │   ├── frame_0000.raw
│           │   └── ...
│           └── lxr/
│
├── Data_for_pyVHR/
│   └── gt_data/
│       ├── gt_hyl/                     # 受试者 hyl 的 GT 数据
│       │   └── bpms_times_GT.npz
│       ├── gt_lj/                      # 受试者 lj 的 GT 数据
│       │   └── bpms_times_GT.npz
│       └── gt_lxr/
│
└── automation_config.yaml
    pyvhr:
      subjects: ["hyl", "lj", "lxr"]    # 与目录名一致
```

---

## 常见错误和解决方案

### ❌ 错误 1：目录名不一致
```
RAW 目录：raw_hyl/
配置文件：subjects: ["hyl"]
结果：找不到视频文件 hyl_output_8bit.mkv
```

**解决**：将目录重命名为 `hyl/`

### ❌ 错误 2：GT 目录缺少前缀
```
GT 目录：hyl/
配置文件：subjects: ["hyl"]
结果：找不到 GT 文件
```

**解决**：将目录重命名为 `gt_hyl/`

### ❌ 错误 3：配置文件中的 subject 拼写错误
```
RAW 目录：hyl/
GT 目录：gt_hyl/
配置文件：subjects: ["hly"]  # 拼写错误
结果：映射失败
```

**解决**：修正配置文件为 `subjects: ["hyl"]`

---

## 快速开始

### 1. 运行自动化流程

```bash
python automation_pipeline.py --config automation_config.yaml
```

### 2. 修改配置文件

编辑 `automation_config.yaml`，修改要扫描的参数：

```yaml
isp:
  parameter_sweep:
    target_module: "gammacorrection"  # 模块名
    target_param: "gamma"             # 参数名
    values: [0.8, 1.0, 2.2, 3.0]      # 扫描值
```

## 支持的参数扫描

### 数值型参数
- `gammacorrection.gamma`: [0.8, 1.0, 2.2, 3.0]
- `contrastsaturation.contrast_factor`: [0.8, 1.0, 1.2]
- `contrastsaturation.saturation_factor`: [0.8, 1.0, 1.2]
- `sharpen.amount`: [0.5, 1.0, 1.5]

### 算法选择型参数
- `demosaic.algorithm`: ["bilinear", "malvar2004", "AHD"]
- `whitebalance.algorithm`: ["gray_world_green", "fixed_gain"]
- `colorcorrectionmatrix.method`: ["sensor_to_srgb", "identity"]

## 输出结构

```
Data_for_pyVHR/isp_output_Video/
└── sensitivity_analysis/baseenv/gamma/
    ├── 0.8/raw_hyl_output_8bit.mkv
    ├── 1.0/raw_hyl_output_8bit.mkv
    └── ...

rPPGanalyze_res_plots/sensitivity_analysis/gamma/cpu_CHROM_v1/
├── group_1_baseenv_0.8gammaISP_VG/
│   ├── singleVideo_rppg_analysis/
│   ├── group_summary.json
│   └── group_comparison.png
├── all_groups_summary.json
├── all_groups_barchart_of_average_errors.png
└── all_groups_metrics_boxplot.png
```

## 独立运行原有脚本

### 单独运行 ISP
```bash
cd ISPpipline
python ISP_main.py
```

### 单独运行 pyVHR 分析
```bash
cd pyVHR_run_on_video
python analyze_with_pyvhr.py
```

## 工作流程

```
配置文件 → 参数变体生成
    ↓
For each 参数值:
    ├─ ISP 处理所有受试者 (11 个视频)
    ├─ 构建 video_group
    └─ pyVHR 分析 (计算组内平均误差)
    ↓
跨组对比 (生成条形图和箱型图)
```

## 示例：Gamma 消融实验

```yaml
experiment_name: "gamma_ablation_study"
isp:
  parameter_sweep:
    target_module: "gammacorrection"
    target_param: "gamma"
    values: [0.8, 1.0, 2.2, 3.0]
```

运行后自动完成：
- ISP 处理：4 组参数 × 11 个受试者 = 44 个视频
- pyVHR 分析：4 个 video_groups
- 生成对比图表和 JSON 摘要

## 注意事项

1. 确保 FFmpeg 已安装并在 PATH 中
2. 确保 GT 数据文件存在于 `Data_for_pyVHR/gt_data/`
3. 确保 RAW 视频帧存在于 `ISPpipline/raw_data/baseenv_rawframe/`
4. 首次运行建议使用较少的参数值进行测试
