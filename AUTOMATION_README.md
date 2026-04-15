# ISP-pyVHR 自动化集成系统使用说明

## ⚠️ 重要：目录结构和命名规范

### 核心映射规则

自动化系统通过**目录名称**进行映射，必须确保以下对应关系：

```
配置文件中的 subject = RAW 目录名
GT 目录名 = gt_{subject}
```

#### 输入文件的命名规范

1. **RAW 视频帧目录**：`raw_data/baseenv_rawframe/{任意名称}/`
2. **GT 数据目录**：`Data_for_pyVHR/gt_data/gt_{任意名称}/`
3. **配置文件**：`subjects: ["{任意名称}"]`

**关键**：三者中的 `{任意名称}` 是对应的，必须完全一致。

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

### `steps` 级联参数扫描

对于 `rawdenoise.steps` 这类级联结构，`target_param` 支持嵌套路径写法，而不再局限于模块下的一级字段。

支持的写法：
- 一级字段：`gamma`
- 按列表索引：`steps[0].alpha` 或 `steps.0.alpha`
- 按算法名匹配步骤：`steps[temporal].alpha`
- 更通用的字段匹配：`steps[algorithm=temporal].alpha`

#### 示例 1：扫描 temporal 的 `alpha`

```yaml
isp:
  baseline_params:
    rawdenoise:
      enabled: true
      bayer_pattern: "GRBG"
      steps:
        - algorithm: "temporal"
          alpha: 0.6
          motion_thresh: 0.02

  parameter_sweep:
    target_module: "rawdenoise"
    target_param: "steps[temporal].alpha"
    values: [0.3, 0.5, 0.7, 0.9]
```

#### 示例 2：扫描 spatial 降噪参数

```yaml
isp:
  baseline_params:
    rawdenoise:
      enabled: true
      bayer_pattern: "GRBG"
      steps:
        - algorithm: "temporal"
          alpha: 0.6
        - algorithm: "bilateral"
          sigma: 1.2

  parameter_sweep:
    target_module: "rawdenoise"
    target_param: "steps[bilateral].sigma"
    values: [0.8, 1.2, 1.6, 2.0]
```

#### 使用建议

- 如果同一个 `steps` 列表里某种 `algorithm` 只出现一次，推荐写 `steps[temporal].alpha`，更直观。
- 如果同一种 `algorithm` 出现多次，推荐写 `steps[0].alpha` 这类索引形式，避免匹配到错误步骤。
- 扫描结果的目录维度名会自动使用叶子参数名，例如 `steps[temporal].alpha` 会生成 `.../alpha/<value>/...`。

## 输出目录结构

下面所有路径均按照“固定前缀 + 配置字段 + 自动拼接变量”的方式生成。

### 1. ISP 输出视频

通式：

```text
参数扫描时：
<isp_video_base>/<experiment_type>/baseenv/<experiment_name>/<path_param_dim>/<variant_name>/<video_name>_output_<output_bit_depth>bit.mkv

不扫描时：
<isp_video_base>/<experiment_type>/baseenv/<path_param_dim>/<variant_name>/<video_name>_output_<output_bit_depth>bit.mkv
```

其中：
- `<isp_video_base>` 来自 `automation_config.yaml` 中的 `output.isp_video_base` 配置项，例如 `Data_for_pyVHR/isp_output_Video`

- `<experiment_type>` 来自 `automation_config.yaml` 顶层的 `experiment_type` 配置项，例如 `sensitivity_analysis`

- `baseenv` 是当前代码里的固定路径段

- `<experiment_name>` 来自 `automation_config.yaml` 顶层的 `experiment_name` 配置项，仅在参数扫描时拼接到 ISP 视频/帧路径中

- `<path_param_dim>` 为参数维度名
- 扫描时，`<path_param_dim>` 取 `parameter_sweep.target_param` 的叶子参数名，例如 `steps[temporal].alpha` 会变成 `alpha`
- 不扫描时，`<path_param_dim>` 取 `experiment_name`

- `<variant_name>` 为当前参数值，例如 `0.4`、`0.5`；不扫描时取 `output.single_run_dirname`

- `<video_name>` 为每个视频目录名，例如 `hyl`、`lzz`

- `<output_bit_depth>` 来自 `automation_config.yaml` 中的 `isp.output_bit_depth` 配置项，例如 `8`

示例：

```text
Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/rawdenoise_study/alpha/0.4/hyl_output_8bit.mkv
```

### 2. ISP 输出帧

通式：

```text
参数扫描时：
<isp_frame_base>/<experiment_type>/baseenv/<experiment_name>/<path_param_dim>/<variant_name>/<video_name>/frame_XXXX.png

不扫描时：
<isp_frame_base>/<experiment_type>/baseenv/<path_param_dim>/<variant_name>/<video_name>/frame_XXXX.png
```

其中：
- `<isp_frame_base>` 来自 `automation_config.yaml` 中的 `output.isp_frame_base` 配置项，例如 `ISPpipline/isp_output_frame`
- 其他字段含义与 ISP 视频输出完全一致
- `<video_name>` 这一层是每个受试者的视频目录，例如 `hyl`

示例：

```text
ISPpipline/isp_output_frame/sensitivity_analysis/baseenv/rawdenoise_study/alpha/0.4/hyl/frame_0001.png
```

### 3. 探针输出结果

通式：

```text
<save_dir>/<probes_output_dirname>/<probes_output_subdir>/<video_name>/<probe_name>/
```

其中：
- `<save_dir>` 来自 `automation_config.yaml` 中的 `isp.probe_system.save_dir` 配置项，例如 `probes_debug`

- `<probes_output_dirname>` 来自 `automation_config.yaml` 中的 `output.probes_output_dirname` 配置项，例如 `probes_experiment`

- `<probes_output_subdir>` 来自 `automation_config.yaml` 中的 `output.probes_output_subdir` 配置项，例如 `spatial_algorithm_scan`

- `<video_name>` 为每个视频名，例如 `lzz`、`hyl`

- `<probe_name>` 是探针位置名称，例如 `Input`、`After_DPC`、`After_RawDenoise`

注意：
- `probes_debug` 不是写死的名字，而是来自 `automation_config.yaml` 中的 `isp.probe_system.save_dir`
- 为避免同一个探针实验目录下的不同运行互相覆盖，建议总是配置 `<probes_output_subdir>`
- 如果未配置 `output.probes_output_subdir`，则会回退到旧路径：`<save_dir>/<probes_output_dirname>/<video_name>/`

示例：

```text
probes_debug/probes_experiment/spatial_algorithm_scan/hyl/Input/
probes_debug/probes_experiment/spatial_algorithm_scan/hyl/After_DPC/
probes_debug/probes_experiment/spatial_algorithm_scan/hyl/After_RawDenoise/
```

### 4. pyVHR 总结果目录

通式：

```text
参数扫描时：
<analysis_results_base>/<experiment_type>/<experiment_name>/<path_param_dim>/<rppg_method>/

不扫描时：
<analysis_results_base>/<experiment_type>/<path_param_dim>/<rppg_method>/
```

其中：
- `<analysis_results_base>` 来自 `automation_config.yaml` 中的 `output.analysis_results_base` 配置项，例如 `rPPGanalyze_res_plots`
- `<experiment_type>` 来自 `automation_config.yaml` 顶层的 `experiment_type` 配置项
- `<experiment_name>` 来自 `automation_config.yaml` 顶层的 `experiment_name` 配置项，仅在参数扫描时拼接到 pyVHR 结果路径中
- `<path_param_dim>` 与 ISP 输出中的规则相同
- `<rppg_method>` 来自 `automation_config.yaml` 中的 `pyvhr.analysis_params.rppg_method` 配置项，例如 `cpu_CHROM`

示例：

```text
rPPGanalyze_res_plots/sensitivity_analysis/rawdenoise_study/alpha/cpu_CHROM/
```

### 5. pyVHR 分组结果

在 pyVHR 总结果目录下，会继续生成每个视频组的子目录：

```text
参数扫描时：
<analysis_results_base>/<experiment_type>/<experiment_name>/<path_param_dim>/<rppg_method>/group_<idx>_<group_name>/

不扫描时：
<analysis_results_base>/<experiment_type>/<path_param_dim>/<rppg_method>/group_<idx>_<group_name>/
```

其中：
- `<idx>` 是自动编号，如 `1`、`2`
- `<group_name>` 是自动构建的视频组名
- baseline 组固定为 `baseenv_baselineISP_VG`
- 扫描实验组通常为 `baseenv_<variant_name><param_label>ISP_VG`

组目录下的典型文件结构：

```text
group_1_baseenv_baselineISP_VG/
├── singleVideo_rppg_analysis/
├── group_summary.json
└── group_comparison.png
```

其中单视频结果图位于：

```text
group_<idx>_<group_name>/singleVideo_rppg_analysis/<video_name>_heartRate_comparison.png
```

### 6. pyVHR 全局汇总文件

在 pyVHR 总结果目录下，还会生成：

```text
all_groups_summary.json
all_groups_barchart_of_average_errors.png
all_groups_metrics_boxplot.png
```

### 7. 固定 baseline 视频组路径

无论是否进行参数扫描，自动化流程在 pyVHR 分析阶段都会额外加载一个固定 baseline 视频组：

```text
Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/<video_name>_output_<output_bit_depth>bit.mkv
```

其中：
- `Data_for_pyVHR/isp_output_Video/baseenv_baselineISP` 是当前代码里的固定路径
- `<video_name>` 为视频名，例如 `hyl`
- `<output_bit_depth>` 来自 `automation_config.yaml` 中的 `isp.output_bit_depth` 配置项

示例：

```text
Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/hyl_output_8bit.mkv
```

### 无参数扫描时自定义最后一级目录名

如果没有配置 `parameter_sweep`，则：
- ISP 视频与 ISP 帧中的 `<path_param_dim>` 使用 `experiment_name`
- `<variant_name>` 使用 `output.single_run_dirname`

你可以在配置文件中自行指定：

```yaml
output:
  isp_video_base: "Data_for_pyVHR/isp_output_Video"
  isp_frame_base: "ISPpipline/isp_output_frame"
  analysis_results_base: "rPPGanalyze_res_plots"
  single_run_dirname: "binning_average"
```

此时目录会变为：

```text
Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/rawdenoise_study/binning_average/
ISPpipline/isp_output_frame/sensitivity_analysis/baseenv/rawdenoise_study/binning_average/
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

---

## ISP 模块开关控制

### 启用/禁用模块

每个 ISP 模块都支持 `enabled` 字段来控制是否执行：

```yaml
isp:
  baseline_params:
    blacklevelcorrection:
      enabled: true   # 启用黑电平校正
      black_level: 0
    
    rawdenoise:
      enabled: false  # 禁用 RAW 域降噪
      steps: [...]
    
    sharpen:
      enabled: false  # 禁用锐化
```

## 探针输出目录

自动化运行时，探针目录由三级配置控制：

```yaml
isp:
  probe_system:
    save_dir: "probes_debug"

output:
  probes_output_dirname: "probes_experiment"
  probes_output_subdir: "spatial_algorithm_scan"
```

对应输出路径为：

```text
probes_debug/<probes_output_dirname>/<probes_output_subdir>/<video_name>/
```

例如当：
- `output.probes_output_dirname: "probes_experiment"`
- `output.probes_output_subdir: "spatial_algorithm_scan"`

时，路径为：

```text
probes_debug/probes_experiment/spatial_algorithm_scan/hyl/
probes_debug/probes_experiment/spatial_algorithm_scan/ycl/
```

### 模块分类

**必需模块**（建议保持 `enabled: true`）：
- `demosaic`：Bayer 去马赛克
- `colorspaceconversion`：RGB ↔ YUV 转换
- `yuvtorgb`：YUV → RGB 转换

**可选模块**（可根据需要开关）：
- `blacklevelcorrection`：黑电平校正
- `defectpixelcorrection`：坏点校正
- `rawdenoise`：RAW 域降噪
- `whitebalance`：白平衡
- `colorcorrectionmatrix`：色彩校正矩阵
- `gammacorrection`：Gamma 校正
- `denoise`：RGB 域降噪
- `sharpen`：锐化
- `contrastsaturation`：对比度/饱和度调整

### 使用示例

#### 示例 1：最小 ISP 流水线
```yaml
baseline_params:
  blacklevelcorrection:
    enabled: false
  demosaic:
    enabled: true      # 必需
  colorspaceconversion:
    enabled: true      # 必需
  yuvtorgb:
    enabled: true      # 必需
```

#### 示例 2：验证某个模块的影响

创建两个配置文件对比：

**config_with_sharpen.yaml**：
```yaml
sharpen:
  enabled: true
  amount: 1.2
```

**config_without_sharpen.yaml**：
```yaml
sharpen:
  enabled: false
```

分别运行后对比 rPPG 结果。
