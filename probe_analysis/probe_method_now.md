下面是当前探针分析系统的方案梳理。结论先说：这套系统目前不是在探针 CSV 上直接跑 POS / CHROM / GREEN 这类 pyVHR rPPG 算法，而是把 ISP 各阶段探针导出的 ROI 时序当作“生理代理信号”，用时域、频域、HRV、SQI 和两探针相似度指标来判断各 ISP 模块是否保留或破坏 rPPG 相关成分。

总体流程
当前系统分两层：

探针采集层：PipelineProbe 在 ISP 管线不同位置截取帧数据，基于人脸皮肤 ROI 统计通道均值、帧间 AC 变化、RAW Bayer 子通道等，写出 *_timeseries.csv 和 probe_meta.json。核心在 pipeline_probe.py (line 273)。

探针分析层：batch_analysis.py 按固定 ISP 阶段配对读取两个探针 CSV，调用 run_analysis() 做两两对比。核心配对在 batch_analysis.py (line 29)，核心分析在 probe_ana_cmp.py (line 1023)。

批处理当前按这些阶段比较：

Input -> BlackLevel-DefectPixel
BlackLevel-DefectPixel -> DefectPixel-WhiteBalance
DefectPixel-WhiteBalance -> WhiteBalance-Demosaic
WhiteBalance-Demosaic -> Demosaic-CCM
Demosaic-CCM -> CCM-Gamma
CCM-Gamma -> Gamma-ColorSpace
Gamma-ColorSpace -> ColorSpace-ContrastSaturation
ColorSpace-ContrastSaturation -> ContrastSaturation-YUVtoRGB
ContrastSaturation-YUVtoRGB -> Output
不过要注意，automation_config.yaml 当前只打开了 input / after_demosaic / after_ccm 三个探针，其他阶段是关闭的，所以实际跑 batch_analysis.py 时，缺失的 CSV 会被跳过。配置位置在 automation_config.yaml (line 40)。

ROI 与数据来源
探针采集不是全图统计，而是皮肤 ROI 统计。ROI 生成逻辑如下：

对当前帧构造 reference_rgb：
RAW 帧：按 Bayer pattern demosaic 成 RGB 参考图。
YUV 帧：按 bt709 / bt601 / bt2020 转回 RGB 参考图。
RGB 帧：直接归一化到 uint8 RGB。
用 MediaPipe FaceMesh 检测人脸 landmark。
对全脸做 ConvexHull。
从脸部 mask 中排除眼睛和嘴部区域。
再按亮度阈值去掉过暗、过曝皮肤像素。
同一物理帧的多个 probe 复用同一个 ProbeSkinContext，保证跨探针 ROI 一致性。
如果当前帧检测失败，允许短时间复用上一帧有效 mask，当前配置是最多 3 帧。
对应代码在 pipeline_probe.py (line 10)、pipeline_probe.py (line 182)、pipeline_probe.py (line 214)。

CSV 基础字段：

Frame_ID
Global_Mean
Global_AC_Delta
ROI_Mean_C0
ROI_Mean_C1
ROI_Mean_C2
ROI_AC_Delta
ROI_Valid_Pixel_Count
ROI_Skin_Pixel_Count
ROI_Mask_Source
三通道帧中，ROI_Mean_C0/C1/C2 就是 ROI 内三个通道均值。单通道 RAW 旧逻辑中，ROI_Mean_C0 是混合 RAW 像素均值，C1/C2 为 N/A。

RAW Bayer-Aware 数据
当前代码已经支持 RAW Bayer-aware 字段。RAW 的 Bayer pattern 支持：

RGGB
BGGR
GRBG
GBRG
当前配置中 sensor Bayer pattern 是 GRBG，见 automation_config.yaml (line 20)。

RAW Bayer-aware 统计的字段包括：

RAW_R_Mean
RAW_G1_Mean
RAW_G2_Mean
RAW_G_Mean
RAW_B_Mean
RAW_R_Count
RAW_G1_Count
RAW_G2_Count
RAW_B_Count
RAW_R_AC_Delta
RAW_G1_AC_Delta
RAW_G2_AC_Delta
RAW_G_AC_Delta
RAW_B_AC_Delta
RAW 子通道划分是基于原图坐标 (y % 2, x % 2)，不是裁 ROI 后再局部取奇偶，这一点比较正确，可以避免 ROI 左上角变化导致 CFA 分类错位。实现见 pipeline_probe.py (line 284)、pipeline_probe.py (line 391)、pipeline_probe.py (line 408)。

分析前处理
两个探针 CSV 的分析流程：

读取两个 CSV。
所有列转 numeric，N/A 转 NaN。
按 Frame_ID inner join 对齐。
按时间排序。
对缺失值做线性插值。
自动判断每个探针属于 raw / rgb / yuv。
按域选择主分析通道。
对主通道做预处理：
3 点中值滤波去尖刺。
线性 detrend 去慢漂。
保留 DC 均值。
部分指标使用去 DC 信号，部分指标使用带通滤波信号。
对齐和预处理代码在 probe_ana_cmp.py (line 83)、probe_ana_cmp.py (line 114)。

带通滤波参数：

Butterworth
4 阶
频段 0.7 - 2.0 Hz
对应心率约 42 - 120 BPM
实现见 probe_ana_cmp.py (line 99)。

代码与文档已统一为 0.7-2.0 Hz（HR 候选频段）。

域判定与主通道选择
自动域判定逻辑：

如果 probe_meta.json 中 raw_mode == bayer_aware 或存在 raw_bayer_pattern，判为 raw。
Output 或 *-Output 判为 rgb。
目录名以 colorspace- 或 contrastsaturation- 开头，判为 yuv。
如果 ROI_Mean_C1 全是 NaN，判为 raw。
其他情况判为 rgb。
代码见 probe_ana_cmp.py (line 64)。

主分析通道选择：

域	优先主通道	回退通道	含义
RAW	RAW_G_Mean	ROI_Mean_C0	优先用 Bayer-aware 绿色通道；旧数据回退混合 RAW
RGB	ROI_Mean_C1	ROI_Mean_C0, ROI_Mean_C2	优先绿色 G 通道
YUV	ROI_Mean_C0	ROI_Mean_C1, ROI_Mean_C2	优先亮度 Y 通道
代码见 probe_ana_cmp.py (line 738)、probe_ana_cmp.py (line 763)。

也就是说，当前主线分析的核心通道是：

RAW：RAW_G_Mean，如果没有新版 RAW 字段则用 ROI_Mean_C0
RGB：ROI_Mean_C1，即 G 通道
YUV：ROI_Mean_C0，即 Y/Luma 通道
核心算法与指标
当前分析不是单一算法，而是一组信号质量和跨阶段保真度指标。

时域指标
代码见 probe_ana_cmp.py (line 159)。

输出：

Mean：主通道均值。
Std：主通道标准差。
AC_PeakToPeak：去 DC 后 AC 峰峰值。
SNR_dB：基于 Welch PSD 的频域峰值 SNR。
AC_Delta_Mean：帧间 AC delta 均值。
AC_Delta_Std：帧间 AC delta 标准差。
AC_DC_Ratio：原始主通道标准差 / 原始主通道均值。
SNR 算法
代码见 probe_ana_cmp.py (line 133)。

当前 SNR 计算方式：

对去 DC 信号做 Welch PSD。
在 0.7 - 2.0 Hz 心率候选频段找最大峰值频率 f_peak。
信号能量：f_peak ± 0.2 Hz 内 PSD 积分。
噪声能量：0 - 4 Hz 内除峰值频带外的 PSD 积分。
SNR_dB = 10 * log10(signal_power / noise_power)。
这个是目前最核心的质量指标，表示心率主频相对背景噪声是否突出。

频域指标
代码见 probe_ana_cmp.py (line 175)。

输出：

HR_Band_Ratio：0.7 - 2.0 Hz 能量 / 全频 PSD 能量。
BPM_Estimate：心率频段最大 PSD 峰值频率乘以 60。
Peak_Freq_Hz：峰值频率。
Harmonic_Ratio：主峰 PSD / 二次谐波位置 PSD。
Spectral_Entropy：心率频段内 PSD 归一化后的谱熵。
HRV 指标
代码见 probe_ana_cmp.py (line 210)。

流程：

对主通道做 0.7 - 2.0 Hz 带通。
find_peaks() 找峰。
峰间距转换成 RR interval，单位 ms。
计算：
SDNN
RMSSD
pNN50
LF_HF_Ratio
LF_HF_Ratio 的做法是把 RR 序列插值到 4 Hz，再用 Welch PSD 计算：

LF：0.04 - 0.15 Hz
HF：0.15 - 0.4 Hz
这部分应作为辅助指标。因为它没有外部心率或 PPG ground truth 校验，峰检测错误会直接污染 HRV。

SQI 信号质量指标
代码见 probe_ana_cmp.py (line 273)。

输出：

Peak_Prominence_Ratio：带通信号峰突出度 / 带通信号标准差。
Autocorr_Peak：自相关峰值，反映周期性。
Zero_Crossing_Rate：AC 信号过零率。
SampleEntropy：样本熵，反映信号复杂度或不规则性。
两探针对比指标
代码见 probe_ana_cmp.py (line 306)。

输出：

PCC：两探针主通道 Pearson 相关系数。
Phase_Delay_ms：互相关最大值对应的时延。
DTW_Distance：名称叫 DTW，但当前实现不是严格 DTW，而是两个 z-score 信号的逐点 RMS 差。
BPM_Diff：探针 2 BPM - 探针 1 BPM。
Delta_SNR_dB：探针 2 SNR - 探针 1 SNR。
Spectral_Cosine_Sim：两个 PSD 曲线的余弦相似度。
跨域比较时，例如 RAW -> RGB、RGB -> YUV、YUV -> RGB，会先把两路主信号 z-score 标准化，再算对比指标，避免量纲差异直接主导结果。实现见 probe_ana_cmp.py (line 1111)。

RAW 多通道分析
如果 RAW CSV 中存在新版 Bayer-aware 字段，系统会额外做 RAW 多通道分析。RAW 通道列表在 probe_ana_cmp.py (line 38)。

参与 RAW 多通道分析的通道：

RAW_R
RAW_G1
RAW_G2
RAW_G
RAW_B
每个 RAW 子通道都会计算一套单通道指标，包括：

Mean
Std
AC_PeakToPeak
SNR_dB
AC_Delta_Mean
AC_Delta_Std
HR_Band_Ratio
BPM_Estimate
Peak_Freq_Hz
Harmonic_Ratio
Spectral_Entropy
SDNN
RMSSD
pNN50
LF_HF_Ratio
Peak_Prominence_Ratio
Autocorr_Peak
Zero_Crossing_Rate
SampleEntropy
AC_DC_Ratio
但最终 metrics_summary.csv 中重点展示的是：

每个 RAW 子通道的 SNR_dB
BPM_Estimate
AC_DC_Ratio
Autocorr_Peak
汇总逻辑见 probe_ana_cmp.py (line 903)。

RAW 内部一致性指标重点看：

RAW_G1_G2_PCC
RAW_G1_G2_Phase_Delay_ms
RAW_G1_G2_DTW_Distance
RAW_G1_G2_BPM_Diff
RAW_G1_G2_Delta_SNR_dB
RAW_G1_G2_Spectral_Cosine_Sim
RAW_G1_G2_AC_Delta_Mean_Diff
RAW_R_RAW_G_Corr
RAW_B_RAW_G_Corr
RAW_R_RAW_B_Corr
这部分主要用于判断 RAW Bayer 通道是否合理，尤其是 G1/G2 理论上应当高度一致。实现见 probe_ana_cmp.py (line 491)、probe_ana_cmp.py (line 922)。

输出结果
每一组探针对比会生成：

metrics_summary.csv：分区汇总指标。
timeseries_comparison.png：三个 ROI 通道的带通时域波形。
psd_comparison.png：主通道 PSD 对比，标出 HR band、峰值 BPM、SNR。
spectrogram.png：两个探针主通道时频图。
hrv_poincare.png：RR interval Poincare 图。
raw_channels/timeseries_*.png：RAW 多通道带通波形。
raw_channels/psd_*.png：RAW 多通道 PSD。
raw_channels/g1g2_consistency_*.png：RAW G1/G2 一致性。
raw_channels/corr_heatmap_*.png：RAW 通道相关性热图。
可视化调用位置在 probe_ana_cmp.py (line 1137)。