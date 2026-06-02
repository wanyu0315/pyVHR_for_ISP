# 3. ISP 模块专属分析框架

> 本文件是 experiment-analysis Skill 第 3 节的完整内容，按需加载。
> 分析涉及某个具体 ISP 模块时，必须先读对应小节，再下机理结论；不可套用通用模板（如"看 SNR 涨跌"就完事）。
> 每个 ISP 模块都有**自己的算法假设**和**自己对 rPPG 的影响通道**，分析时必须对号入座。

## 3.1 BlackLevelCorrection（BLC）

- **数学性质**：纯 DC 平移
- **必须验证**：AC_PeakToPeak 与 Compare_PCC 应几乎不变；若变化 → 实现有 bug 或 RAW clipping
- **rPPG 影响通道**：理论上无影响；但若 BL 设置过高导致 clipping，会**抑制低光皮肤区域的 AC**
- **创新候选**：自适应 BL（按 ROI 而非全图估计），保护低光像素 AC

## 3.2 DefectPixelCorrection（DPC）

- **数学性质**：局部中值/邻域插值
- **必须验证**：ROI 内 Compare_PCC > 0.99；若 ROI 内坏点比例 > 1% → DPC 强度需要看
- **rPPG 影响通道**：坏点本身是高方差噪声源；过强 DPC 会"抹平"皮肤纹理上的真实 AC
- **关注指标**：AC_Delta_Std、Zero_Crossing_Rate 异常下降
- **创新候选**：ROI-aware DPC（皮肤区域使用更保守阈值）

## 3.3 RawDenoise（RAW 域降噪）

- **数学性质**：通常 NLM / bilateral / 简单高斯
- **核心矛盾**：rPPG AC 幅度仅约 0.1–1% 量级，与噪声同量级 → **降噪极易把信号一起抹掉**
- **必看**：
  - `Compare_Delta_SNR_dB`（关键）：负值意味着降噪在损害信号
  - HR_Band_Ratio：降噪后是否被高频噪声拉低
  - alpha / binning 等强度扫描的单调性
- **预期四象限**：弱降噪 → ideal_enhance；强降噪 → over_idealize 或 pure_harm
- **创新候选**：**频带选择性降噪**（保留 0.65–4.0 Hz 心率频带、压制其他频带），或**时序降噪而非空间降噪**

## 3.4 WhiteBalance（WB）

- **数学性质**：通道增益（静态：固定增益；动态：gray_world / 等）
- **最危险模块之一**：动态 AWB 每帧增益不同 → 在通道间引入**伪 AC**，掩盖真实心跳节律
- **必看**：
  - 固定增益（fixed_gain）→ 期望 Compare_PCC ≈ 1
  - 动态增益（gray_world）→ 重点看 BPM_Estimate 是否漂移、HR_Band_Ratio 是否塌方
  - G_R_Ratio、G_B_Ratio 的方差（应**接近常数**；若每帧波动大 → AWB 不稳定）
- **跨受试者**：肤色不同对 AWB 触发不同，N ≥ 10 必须分肤色档位看
- **创新候选**：**时间常数自适应 AWB**（增益的时间平滑窗口 >> 心率周期，避免污染 0.65–4.0 Hz 频带）

## 3.5 Demosaic

- **数学性质**：CFA 插值（bilinear / malvar2004 / AHD / 其他）
- **核心影响**：插值在 R/G/B 间引入串扰，改变 CHROM/POS 假设的"皮肤色度比"
- **必看**：
  - 各算法的 Compare_Delta_SNR_dB 对比
  - RAW_G1_G2_PCC（demosaic 前应该接近 1）
  - 在 RGB 探针上：G_R_Ratio / G_B_Ratio 的**方差变化**
  - Layer1 锚 B 的四象限（重点看 CHROM/POS，因为它们对通道串扰最敏感）
- **典型预期**：高质量 demosaic（AHD/malvar2004）应优于 bilinear；但若过度边缘锐化会破坏皮肤区域的低频 AC
- **创新候选**：**rPPG 友好 demosaic**（在皮肤 ROI 内用更保守插值，避免锐化生理信号频带）

## 3.6 CCM（Color Correction Matrix）

- **数学性质**：3×3 线性矩阵变换
- **rPPG 影响通道**：直接改变 RGB 三通道比例 → 改变 CHROM/POS 投影方向 → 改变 BVP 信号方向
- **必看**：
  - identity vs sensor_to_srgb 的对比，特别看 CHROM/POS 的 Layer1 SNR_dB
  - 频谱形状（Spectral_Cosine_Sim）是否保留——CCM 是线性的，理论上**频谱形状应当保持**
- **典型反直觉发现**：identity CCM（不做色彩校正）**可能**对 rPPG 更友好——因为 sensor 通道更接近原始光谱响应
- **创新候选**：**rPPG-optimized CCM**（在已知 GT 心率的训练集上，反向优化 CCM 使 SNR 最大化）→ 这是论文级别的创新

## 3.7 Gamma

- **数学性质**：非线性幂函数 `out = in ^ (1/γ)`（或 `in ^ γ`，看实现方向）
- **rPPG 影响通道**：
  - 不破坏频率位置（峰值频率不变）
  - **改变 AC/DC 比**（低光区域 γ < 1 会放大 AC，但也放大噪声；高光区域 γ > 2 会压缩 AC）
  - 非线性使 CHROM/POS 的线性假设失效（小幅度时近似线性，大动态范围下偏差大）
- **必看**：
  - HR_Band_Ratio（频率信息）应保持
  - AC_DC_Ratio 应有明显趋势变化
  - 跨 γ 扫描时 BPM_MAE 的**最低点**位置
- **当前观察**：γ ∈ {0.8, 1.0, 2.2, 3.0} 已扫，建议补充 γ < 0.8 看是否仍下降，γ > 3 看是否塌方
- **创新候选**：**ROI 自适应 γ**（皮肤 ROI 使用 γ 接近 1，非皮肤区按显示标准 γ=2.2）

## 3.8 ColorSpaceConversion（RGB↔YUV）

- **数学性质**：纯线性矩阵
- **必须验证**：Compare_PCC ≈ 1（任何偏离 1 都是实现 bug）
- **rPPG 影响通道**：Y 通道集中亮度信息，U/V 包含色度——pyVHR 默认在 RGB 域工作，YUV 中分析需要重新映射
- **创新候选**：直接在 YUV 域跑修改版 CHROM（已有文献尝试，可作为本项目对照实验）

## 3.9 ContrastSaturation / Sharpening / YUVDenoise

- **数学性质**：非线性局部增强
- **典型有害行为**：锐化放大高频噪声，对比拉伸将 AC 推到 clipping
- **必看**：Spectral_Entropy 上升（噪声变多）、HR_Band_Ratio 下降
- **创新候选**：**心率频带保护**——这些模块应当显式排除生理信号频段不做增强
