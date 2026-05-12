# 探针系统 GT 信号接口使用报告

## 1. 引入目标

本次引入的是第一种 GT 方案：`Data_for_pyVHR/gt_data` 中的接触式 PPG GT。它的作用不是替代探针系统原有的无参考质量指标，而是在每个 ISP 环节探针分析结果中额外增加“相对于真实生理参考”的失真评估。

引入后，每次执行两个探针 CSV 的对比分析时，如果传入 GT 对象，系统会同时输出：

- 原有无 GT 指标：`SNR_dB`、`BPM_Estimate`、`HR_Band_Ratio`、`PCC`、`Delta_SNR_dB` 等。
- 新增 GT 参考指标：`GT_SNR_dB`、`BPM_MAE`、`BPM_RMSE`、`BPM_Bias`、`BPM_PCC`、`BPM_CCC` 等。
- 新增窗口级明细：`gt_window_metrics.csv`。

这样可以观察每个 ISP 环节处理后，ROI 时序信号中的生理成分是否更接近 GT，或者是否被当前 ISP 环节削弱、偏移、污染。

## 2. 已实现的接口结构

新增文件：

- `probe_analysis/gt_reference.py`

核心接口：

- `BPMGTReference`
- `load_gt_reference(...)`

当前支持的数据源：

- `contact_ppg`：从 `Data_for_pyVHR/gt_data/gt_<subject>/bpms_times_GT.npz` 加载。
- `raw_video_rppg`：已预留文件型入口，后续只要 RAW 视频 rPPG 方案导出同样包含 `bpm` 与 `times` 数组的 `.npz` 文件，就可以通过同一接口接入。

当前 `contact_ppg` GT 文件要求：

- `bpm`：一维 BPM 序列。
- `times`：一维时间戳，单位秒，表示每个 GT BPM 对应的时间。
- `metadata`：可选，当前用于记录采样率、窗口长度等信息。

注意：当前 `bpms_times_GT.npz` 中的 `metadata.source_ppg_file` 路径可能是旧路径，不作为实际加载依据。系统实际按目录规范加载：

```text
Data_for_pyVHR/gt_data/gt_<subject>/bpms_times_GT.npz
```

## 3. 探针分析接入方式

`probe_ana_cmp.py` 的 `run_analysis()` 已新增可选 GT 参数：

```python
run_analysis(
    csv1,
    csv2,
    fps,
    output_dir,
    domain_arg="auto",
    gt_ref=gt_ref,
    gt_window_size=16.0,
    gt_stride=1.0,
    probe_time_mode="absolute_frame_id",
    probe_time_offset_sec=0.0,
)
```

无 GT 时：

- `gt_ref=None`
- 原有分析行为保持不变。

有 GT 时：

- 在 `metrics_summary.csv` 中新增 `--- GT 参考对比区 ---`。
- 在输出目录中新增 `gt_window_metrics.csv`。

命令行示例：

```bash
python probe_analysis/probe_ana_cmp.py \
  probes_debug/probes_experiment/colorcorrectionmatrix/method/d65_standard/yjc/Input/Input_timeseries.csv \
  probes_debug/probes_experiment/colorcorrectionmatrix/method/d65_standard/yjc/After_CCM/After_CCM_timeseries.csv \
  --fps 30 \
  --output /tmp/probe_gt_example \
  --gt-source contact_ppg \
  --gt-subject yjc \
  --gt-root Data_for_pyVHR/gt_data
```

批处理入口 `probe_analysis/batch_analysis.py` 已增加配置：

```python
USE_GT = True
GT_SOURCE = CONTACT_PPG_SOURCE
GT_ROOT = DEFAULT_GT_ROOT
GT_PATH = ""
GT_WINDOW_SIZE = 16.0
GT_STRIDE = 1.0
PROBE_TIME_MODE = "absolute_frame_id"
PROBE_TIME_OFFSET_SEC = 0.0
```

如果某个受试者 GT 加载失败，批处理会给出警告，并降级为无 GT 分析，不中断整个批处理。

## 4. 时间轴对齐方式

探针 CSV 中的 `Frame_ID` 会被转换为秒级时间轴，然后用于寻找 GT BPM。

当前支持两种模式：

- `absolute_frame_id`：`time = Frame_ID / fps + offset`
- `relative_csv`：`time = (Frame_ID - first_Frame_ID) / fps + offset`

默认使用：

```text
absolute_frame_id
```

理由：探针 CSV 中的 `Frame_ID` 通常代表原始视频帧号。如果探针只截取了视频中的一段，例如从第 500 帧开始，那么它在 GT 时间轴上应当从 `500 / 30 = 16.67 s` 附近开始，而不是从 0 秒开始。

如果后续确认某些探针 CSV 的 `Frame_ID` 已经被重置为片段内编号，则应改用：

```text
relative_csv
```

或者通过：

```text
probe_time_offset_sec
```

手动补偿采集同步偏移。

## 5. pyVHR 兼容的窗口设置

新增 GT 指标按窗口计算，默认：

- 窗长：`16.0 s`
- 步长：`1.0 s`
- 频带：`0.65-4.0 Hz`

窗口中心时间的计算方式与 pyVHR 的窗口思想一致：每个窗口用中心时间与 GT BPM 对齐。

注意：如果探针 CSV 短于 `gt_window_size`，则不会产生有效 GT 窗口，GT 指标会为空或 `Valid_GT_Window_Count=0`。这种情况不是计算失败，而是信号长度不足以按当前窗口设置做 pyVHR 风格评估。

## 6. GT_SNR_dB 计算方法

新增的 `GT_SNR_dB` 按 pyVHR `get_SNR()` 的代码逻辑实现，而不是探针系统原来的无 GT 主峰 SNR。

对每个探针窗口：

1. 使用 Welch PSD：
   - 频带：`0.65-4.0 Hz`，与 pyVHR `Welch()` 一致，采用开区间筛选。
   - 输出频率转换为 BPM：`pfreqs = 60 * frequency_hz`。
   - 如果窗口长度小于 256 点：`nperseg = n`，`noverlap = int(0.8*n)`。
   - 如果窗口长度不小于 256 点：`nperseg = 256`，`noverlap = 200`。
   - SNR 使用 pyVHR 动态 `nfft`：

```text
nfft = ceil((60 * 2 * NyquistF) / 0.5)
NyquistF = fps / 2
```

30 fps 时：

```text
nfft = 3600
```

2. 取当前窗口中心时间对应的 GT BPM。为贴近 pyVHR `get_SNR()`，SNR 的 GT BPM 选择使用：

```python
curr_ref = reference_hrs[int(window_center_time)]
```

3. 定义信号功率区域：

```text
GT 主频区域:       GT_BPM ± 12 BPM
GT 一次谐波区域:   2 * GT_BPM ± 12 BPM
```

其中 `12 BPM = 0.2 Hz * 60`，与 pyVHR 代码中的 `interv1 = interv2 = 0.2*60` 一致。

4. 定义噪声功率区域：

```text
0.65-4.0 Hz 频带内，除 GT 主频区域与一次谐波区域以外的所有 PSD bins
```

5. 按 pyVHR 使用 PSD bin 求和，而不是积分：

```text
SPower = sum(PSD[GT主频区域 or GT一次谐波区域])
NPower = sum(PSD[其他频率区域])
GT_SNR_dB = 10 * log10(SPower / NPower)
```

6. 单个探针的 `GT_SNR_dB` 是所有有效窗口 SNR 的均值；同时输出：

- `GT_SNR_Median_dB`
- `GT_SNR_Std_dB`
- `Valid_GT_SNR_Window_Count`

## 7. BPM 误差指标计算方法

每个窗口先用 pyVHR `BVP_to_BPM()` 风格估计探针 BPM：

1. Welch PSD。
2. 频带 `0.65-4.0 Hz`。
3. `nfft = 2048`。
4. 取 PSD 最大峰对应的 BPM。

然后与 GT BPM 对齐。误差指标采用 pyVHR `errors.py` 的时间对齐思想：每个估计时间点找 `timesGT` 中最近的 GT 时间点。

输出指标：

- `BPM_MAE`：`mean(abs(BPM_Estimate - BPM_GT))`
- `BPM_RMSE`：`sqrt(mean((BPM_Estimate - BPM_GT)^2))`
- `BPM_MAXError`：最大绝对误差
- `BPM_Bias`：`mean(BPM_Estimate - BPM_GT)`，正值表示探针估计偏高
- `BPM_PCC`：探针 BPM 与 GT BPM 的 Pearson 相关
- `BPM_CCC`：Lin's Concordance Correlation Coefficient
- `GT_BPM_Mean`、`GT_BPM_Min`、`GT_BPM_Max`
- `Valid_GT_Window_Count`

跨探针比较中同时输出差值：

- `Compare_Delta_GT_SNR_dB = P2_GT_SNR_dB - P1_GT_SNR_dB`
- `Compare_Delta_BPM_MAE = P2_BPM_MAE - P1_BPM_MAE`
- `Compare_Delta_BPM_RMSE = P2_BPM_RMSE - P1_BPM_RMSE`
- `Compare_Delta_BPM_Bias = P2_BPM_Bias - P1_BPM_Bias`

解释：

- `Delta_GT_SNR_dB > 0`：P2 相比 P1 在 GT 心率及其谐波附近的能量占比更高。
- `Delta_BPM_MAE < 0`：P2 相比 P1 的 BPM 估计误差更小。
- `BPM_Bias > 0`：探针估计 BPM 偏高。
- `BPM_Bias < 0`：探针估计 BPM 偏低。

## 8. 输出文件

每个探针对比输出目录中，新增或更新：

```text
metrics_summary.csv
gt_window_metrics.csv
```

`metrics_summary.csv` 中新增：

```text
--- GT 参考对比区 ---
```

`gt_window_metrics.csv` 包含窗口级明细：

- `Probe`
- `Window_Index`
- `Window_Start_Frame`
- `Window_End_Frame`
- `Window_Center_s`
- `GT_BPM_For_Error`
- `GT_BPM_For_SNR`
- `BPM_Estimate`
- `BPM_Error_EstMinusGT`
- `GT_SNR_dB`

## 9. 第二种 GT 方案的预留接口

第二种方案是“使用 RAW 视频直接进行 rPPG 分析得到的生理信息作为 GT”。本次没有实现 RAW 视频 rPPG 的生成流程，但已经预留入口：

```python
load_gt_reference(
    source_type="raw_video_rppg",
    subject="<subject>",
    path="<raw_rppg_gt_file.npz>",
)
```

未来 RAW 视频 rPPG GT 只需要导出同样格式：

```text
bpm   : 1D array
times : 1D array, seconds
metadata : optional dict
```

即可复用现有 `GT_SNR_dB`、`BPM_MAE`、`BPM_RMSE`、窗口级明细与汇总表逻辑。

## 10. 使用注意事项

1. `Data_for_pyVHR/gt_data` 目录中的 USB Camera 时间戳约为 10 fps，这是 GT 采集辅助视频，不代表 ISP 探针视频 FPS。探针分析仍应使用实际 ISP 视频 FPS，例如 30 fps。
2. 当前默认 `GT_WINDOW_SIZE=16.0`，与现有 GT BPM 生成配置一致。如果探针片段短于 16 秒，应缩短窗口或使用更长探针 CSV。
3. 时间同步是 GT 评估的核心风险点。如果 `Frame_ID/fps` 与 GT 时间轴存在整体偏移，应使用 `PROBE_TIME_OFFSET_SEC` 校正。
4. 原有 `SNR_dB` 是无 GT 主峰 SNR；新增 `GT_SNR_dB` 是 pyVHR 风格 GT 参考 SNR。两个指标含义不同，后续论文或实验报告中应分开命名。
