# GTPreprocessing 模块说明

本目录集中放置**探针分析系统的 GT (Ground-Truth) 预处理工具链**，为后续 `probe_ana_cmp.run_analysis` 提供两类参考信号：

- **锚 A：接触式 PPG GT**（来自 `Data_for_pyVHR/gt_data/gt_<subject>/bpms_times_GT.npz`）
- **锚 B：RAW proxy-GT**（从 `raw_proxyGT_data/raw_proxyGT_reference/<subject>/.../raw_input_timeseries.csv` 生成）

两类 GT 对 `run_analysis` 完全透明，共用统一的 `BPMGTReference` 数据结构与 `gt_metrics_for_signal` 计算管线。

---

## 文件一览

| 文件 | 角色 |
|-----|-----|
| `gt_reference.py` | GT 统一加载接口，封装 `BPMGTReference` 数据类 + `load_gt_reference()` 工厂 |
| `multi_algo_bpm.py` | 多算法 rPPG BPM 估计底层公共函数（GREEN/CHROM/POS/LGI） |
| `generate_proxy_gt.py` | 多算法 RAW proxy-GT 生成工具（CLI + Python API） |
| `gt_usage_report.md` | 早期版本的 GT 接入使用报告（历史文档，以本文档为准） |

---

## 1. `gt_reference.py`

### 作用

把不同来源的 GT 统一包装成 `BPMGTReference` 数据对象，给下游 `run_analysis` / `gt_metrics_for_signal` 一个算法无关的接口。不关心 GT 是来自接触式 PPG 还是 RAW 视频 rPPG，后面的指标计算管线都一样。

### 核心对象

**`BPMGTReference`**（dataclass）

| 字段 | 类型 | 说明 |
|------|------|------|
| `subject` | str | 受试者名（例 `"yjc"`） |
| `source_type` | str | `"contact_ppg"` 或 `"raw_video_rppg"` |
| `bpm` | np.ndarray (N,) | BPM 序列 |
| `times` | np.ndarray (N,) | 与 bpm 对齐的时间（秒） |
| `data_path` | str | 原始 .npz 路径 |
| `metadata` | dict | 来源详细信息（采样率、窗长、算法等） |

关键方法：
- `bpm_at(query_times, method)`：按查询时间点返回 GT BPM。`method='nearest'`（探针窗口中心查 GT，对应 pyVHR `getErrors` 的对齐方式）或 `'pyvhr_index'`（对应 pyVHR `get_SNR` 的 `reference_hrs[int(timesES[idx])]` 取法）。
- `describe()`：返回指标汇总用的字典（time 范围、BPM 统计量等）。

### 顶层入口

```python
load_gt_reference(source_type, subject="", gt_root="Data_for_pyVHR/gt_data", path="") -> BPMGTReference
```

| 参数组合 | 加载源 |
|---------|-------|
| `source_type="contact_ppg", subject="yjc"` | `Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT.npz`（锚 A） |
| `source_type="raw_video_rppg", path="<...>.npz"` | 指定 .npz 文件（锚 B，由 `generate_proxy_gt.py` 产出） |

### 输入

- **锚 A**：`Data_for_pyVHR/gt_data/gt_<subject>/bpms_times_GT.npz`
  - 必需字段：`bpm` (1D)、`times` (1D，秒)
  - 可选字段：`metadata` (dict，记录采样率/窗长等)
- **锚 B**：任何符合同样字段规范的 .npz（由 `generate_proxy_gt.py` 产出）

### 输出

`BPMGTReference` Python 对象，可被 `probe_ana_cmp.gt_metrics_for_signal()` 直接使用。

### 后续环节

- `probe_ana_cmp.py` 顶部 import 本文件，`run_analysis(..., gt_ref=...)` 和 `run_analysis(..., gt_ref_proxy_by_algo={...})`（M3 后）都依赖此对象。
- `batch_analysis.py` 在每个受试者入口调用 `load_gt_reference()`，把对象传给 `run_analysis`。

---

## 2. `multi_algo_bpm.py`

### 作用

封装 pyVHR 的 4 种 rPPG 算法（GREEN / CHROM / POS / LGI），对外暴露**统一的滑窗多算法 BPM 估计接口**。本模块是探针分析系统中**算法依赖层的唯一入口**，同时被两处调用：

1. **proxy-GT 生成**（`generate_proxy_gt.py`）：为每种算法在 Input 探针 ROI 时序上独立产出 BPM 序列。
2. **Layer 1 探针指标**（M3 后的 `run_analysis`）：为每个探针位置同步跑多算法，配合 proxy-GT 实现**算法对称性对比**。

数值口径严格对齐 pyVHR（频段 0.65-4.0 Hz、Welch + nfft=2048、窗口中心作时间），避免上下游算法不一致导致的对比偏见。

### 关键常量

```python
SUPPORTED_ALGOS = ("GREEN", "CHROM", "POS", "LGI")
SUPPORTED_DOMAINS = ("raw", "rgb", "yuv")
DEFAULT_WINDOW_SIZE_SEC = 16.0
DEFAULT_STRIDE_SEC = 1.0
```

### 核心函数

#### `compute_multi_algo_bpm(signal, frame_ids, fps, algos=None, domain="rgb", ...)`

**输入**：
| 参数 | 形状/含义 |
|------|---------|
| `signal` | RGB/YUV 域: `[N, 3]`；RAW 单通道: `[N]`；RAW 三通道（Bayer-aware R/G/B 均值）: `[N, 3]` |
| `frame_ids` | `[N]` 帧号数组，与 signal 同长 |
| `fps` | 采样率（帧率） |
| `algos` | 算法列表，None 表示全部。RAW 单通道会自动过滤到仅 `GREEN` |
| `domain` | `"raw"` / `"rgb"` / `"yuv"` |
| `window_size_sec` / `stride_sec` | 滑窗参数（默认 16s / 1s） |
| `do_preprocess` | 是否调用 `probe_ana_cmp.preprocess`（中值 + detrend 保留 DC） |

**域处理规则**：
| 域 | 输入形式 | 可用算法 | 处理 |
|----|---------|---------|------|
| `raw` + 单通道 `[N]` | 如 `RAW_G_Mean` | 仅 `GREEN` | G 列放真实信号，R/B 填 0 |
| `raw` + 三通道 `[N, 3]` | `RAW_R/G/B_Mean` | **全 4 种** | 当作 RGB 直通（论文表述为"raw Bayer 响应下的 CHROM/POS"） |
| `rgb` + `[N, 3]` | 常规 RGB 探针 | 全 4 种 | 直通 |
| `yuv` + `[N, 3]` | YUV 探针 | 全 4 种 | BT.709 反变换回 RGB |

**输出**：
```python
{
    "GREEN": {
        "bpm":         np.ndarray (N_win,),    # 各窗口 BPM
        "times":       np.ndarray (N_win,),    # 窗口中心时间（秒）
        "bvp_full":    np.ndarray (N,),        # 整段 BVP（含 DC）
        "bvp_windows": list[np.ndarray],       # 每窗口 BVP 片段
        "ranges":      list[(start, end)],     # 每窗口帧索引
    },
    "CHROM": {...},
    "POS":   {...},
    "LGI":   {...},
}
```

#### `compute_multi_algo_gt_snr(algo_result, fps, gt_bpm_per_window)`

在已有 GT BPM 的情况下，对某算法每个窗口算 pyVHR 风格的 GT-anchored SNR（dB）。**输入**：`compute_multi_algo_bpm` 返回的单算法子字典 + 每窗口对应的 GT BPM 数组。**输出**：`(N_win,)` 的 SNR 数组。主要给 Layer 1 指标做诊断。

#### `load_reference_rgb_from_csv(reference_csv, ...)`

从 `raw_input_timeseries.csv` 中智能挑选可用通道。

**加载优先级**：
1. `RAW_R_Mean` + `RAW_G_Mean` + `RAW_B_Mean` 三列都有效 → `[N, 3]`, domain=`"raw"`（支持全算法）
2. `ROI_Mean_C0/C1/C2` 三列都有效 → `[N, 3]`, domain=`"rgb"`（支持全算法）
3. `RAW_G_Mean` 或 `ROI_Mean_C0` 有效 → `[N]`, domain=`"raw"`（仅 GREEN）
4. 否则抛 `MultiAlgoBpmError`

**输出**：`(signal, frame_ids, domain)` 三元组。

### 后续环节

- `generate_proxy_gt.py` 直接调用 `compute_multi_algo_bpm` 生成 proxy-GT
- M3（`run_analysis` 扩展）时会调用它为每个探针位置同步产出多算法 BPM，作为 Layer 1 指标

---

## 3. `generate_proxy_gt.py`

### 作用

从 **raw_proxyGT_reference CSV**（Input 探针在 RAW 域采集的 ROI 时序）生成**多算法 proxy-GT .npz 文件**。每个算法一份独立 .npz，格式完全兼容 `gt_reference.py` 的 `raw_video_rppg` 源，直接供 `run_analysis` 作锚 B 使用。

**解决的核心问题**：proxy-GT 的 BPM 估计算法必须与后续探针位置的 BPM 估计算法一致（算法对称性），否则对比会被算法偏见污染。因此每种算法各生一份 proxy-GT。

### 核心函数

#### `generate_proxy_gt_from_reference(reference_csv, output_dir, subject, fps=30.0, algos=None, ...)`

**输入**：
| 参数 | 说明 |
|------|------|
| `reference_csv` | `raw_proxyGT_data/raw_proxyGT_reference/<subject>/<hash>/raw_input_timeseries.csv` |
| `output_dir` | `raw_proxyGT_data/proxy_gt/<subject>/`，自动创建 |
| `subject` | 受试者名，决定 .npz 文件名前缀 |
| `fps` | 探针帧率（默认 30） |
| `algos` | 要生成的算法列表，None = 全 4 种 |
| `window_size_sec` / `stride_sec` | 默认 16s / 1s（与接触式 PPG GT 的默认配置一致） |
| `probe_time_mode` / `probe_time_offset_sec` | 时间轴构建参数（默认 `"absolute_frame_id"` + 0 偏移） |
| `overwrite` | 已存在是否覆盖，默认 True |

**输出**：
- 对每个算法写一个 .npz 到 `output_dir/<subject>_proxy_gt_<ALGO>.npz`
- .npz 字段：`bpm` (N_win,)、`times` (N_win,)、`metadata` (object, 含 algo/domain/fps/window_size/reference_csv/bpm_method 等)
- 函数返回值：`{algo: output_npz_path}` 字典

#### `batch_generate_for_root(reference_root, output_root, ...)`

批量对 `reference_root` 下所有 subject 目录生成 proxy-GT。自动发现 `<subject>/*/raw_input_timeseries.csv`，失败的 subject 记 `[ERROR]` 不中断。

### CLI 用法

```bash
# 单受试者
python probe_analysis/GTPreprocessing/generate_proxy_gt.py \
    --reference probe_analysis/raw_proxyGT_data/raw_proxyGT_reference/yjc/start_000300_frames_1600_bayer_GRBG_roi_e195f48be7/raw_input_timeseries.csv \
    --output-dir probe_analysis/raw_proxyGT_data/proxy_gt/yjc \
    --subject yjc --fps 30

# 批量全受试者
python probe_analysis/GTPreprocessing/generate_proxy_gt.py \
    --reference-root probe_analysis/raw_proxyGT_data/raw_proxyGT_reference \
    --output-root   probe_analysis/raw_proxyGT_data/proxy_gt \
    --fps 30

# 只跑某几个算法
python probe_analysis/GTPreprocessing/generate_proxy_gt.py \
    --reference-root probe_analysis/raw_proxyGT_data/raw_proxyGT_reference \
    --output-root   probe_analysis/raw_proxyGT_data/proxy_gt \
    --algos POS CHROM
```

### 输入 / 输出目录规范

```
probe_analysis/raw_proxyGT_data/
├── raw_proxyGT_reference/          # 输入: ISP 管线前置探针采集
│   └── <subject>/
│       └── start_<frame>_frames_<N>_bayer_<pattern>_roi_<hash>/
│           ├── raw_input_timeseries.csv   # 本工具读此文件
│           ├── mask_manifest.csv
│           ├── masks/
│           └── reference_meta.json
└── proxy_gt/                       # 输出: 本工具产物
    └── <subject>/
        ├── <subject>_proxy_gt_GREEN.npz
        ├── <subject>_proxy_gt_CHROM.npz
        ├── <subject>_proxy_gt_POS.npz
        └── <subject>_proxy_gt_LGI.npz
```

### 后续环节

生成的 .npz 在 M3/M4 阶段会被 `batch_analysis.py` 加载并通过 `run_analysis(..., gt_ref_proxy_by_algo={...})` 参数传给探针分析主流程，作为锚 B 参与三锚点指标体系中的「同算法 proxy-GT vs 探针位置 bpmES」对比。

---

## 目录外依赖关系

```
GTPreprocessing/multi_algo_bpm.py
    └── 依赖父目录 probe_analysis/probe_ana_cmp.py 的公共函数：
        preprocess(), pyvhr_bpm_estimate(), pyvhr_gt_snr_db(),
        _pyvhr_welch(), _pyvhr_window_ranges(), build_probe_time_axis()
    └── 依赖 pyVHR.BVP.methods 的 cpu_GREEN/cpu_CHROM/cpu_POS/cpu_LGI

GTPreprocessing/generate_proxy_gt.py
    └── 依赖 GTPreprocessing/multi_algo_bpm.py

GTPreprocessing/gt_reference.py
    └── 独立模块，仅依赖 numpy
```

上层调用入口：
- `probe_analysis/probe_ana_cmp.py` → `from gt_reference import ...`（已配置 sys.path）
- `probe_analysis/batch_analysis.py` → `from gt_reference import ...`（已配置 sys.path）

---

## 常见问题

**Q1：为什么 RAW 三通道可以跑 CHROM/POS？**
CHROM/POS 的数学假设是"皮肤光反射的标准 RGB 投影"。未经 WB/CCM 的 raw Bayer 曝光值近似线性响应，rPPG 研究中有先例直接在这上面做色度分析，只是信号偏见和 sRGB 不同。论文表述为"raw Bayer 响应下的 CHROM/POS"。

**Q2：为什么要为每个算法分别生一份 proxy-GT？**
保证算法对称性。若 proxy-GT 用 GREEN 估计，后续探针又用 POS 估计，则误差里混入了"GREEN vs POS 本身的口径差"，无法干净地归因于 ISP 模块影响。为每个算法独立生成后，同算法内部对比才是 ISP 模块影响的纯净度量。

**Q3：proxy-GT 的窗口数由什么决定？**
由 reference CSV 总长、`window_size_sec` 和 `stride_sec` 决定。对 1600 帧 @ 30fps + 16s/1s → 38 个窗口；`times` 覆盖 17.98s - 54.98s（每个窗口中心）。

**Q4：某个 subject 的 reference CSV 为空会怎样？**
`load_reference_rgb_from_csv` 会抛 `MultiAlgoBpmError`，在 `batch_generate_for_root` 中记 `[ERROR]` 跳过该 subject，不影响其它 subject 的生成。
