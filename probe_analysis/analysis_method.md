RAW 域（Input → BlackLevel → DefectPixel → WhiteBalance）
RAW 域是 Bayer 单通道，只有 C0，没有 RGB 三通道概念。

核心指标：

SNR_dB — 最关键。RAW 域 SNR 是信号质量的上限，后续只会变差不会变好
AC_PeakToPeak — 心跳引起的微弱亮度波动幅度，越大越好
HR_Band_Ratio — 0.7-3.5Hz 能量占比，如果某模块后突然降为 0，说明心率信号被破坏
Autocorr_Peak — 自相关峰值，反映信号周期性是否存在
重点关注：

BlackLevel：应该只平移 DC，不改变 AC 分量。对比前后 AC_PeakToPeak 应几乎不变
DefectPixel：坏点校正不应影响 ROI 区域信号，Compare_PCC 应 > 0.99
WhiteBalance：最危险。如果用动态 AWB（如 gray_world），每帧增益不同，会直接压制心跳信号。看 Compare_Delta_SNR_dB，如果为负值说明 AWB 在损害信号
RGB 域（Demosaic → CCM → Gamma）
Demosaic 后有了 RGB 三通道，可以分析通道间关系。

核心指标：

SNR_dB — 继续追踪，看每个模块的 SNR 变化趋势
BPM_Estimate — 各模块前后估计的心率应一致，如果突变说明信号被扭曲
Spectral_Entropy — 频谱熵越低越好（信号越纯净）。如果某模块后熵突增，说明引入了噪声
HR_Band_Ratio — 心率频段能量占比
重点关注：

Demosaic：插值会引入通道间串扰。对比前后 SNR_dB 下降幅度
CCM：色彩矩阵改变三通道比例。如果 CCM 矩阵不准确，会破坏 POS 算法假设的通道关系。看 Compare_PCC，如果显著低于 1 说明 CCM 改变了信号形态
Gamma：非线性变换。理论上不应破坏频率信息，但会改变 AC/DC 比。看 HR_Band_Ratio 是否保持
YUV 域（ColorSpace → ContrastSaturation → YUVtoRGB）
RGB→YUV 转换后，亮度(Y)和色度(U/V)分离。rPPG 信号主要在色度分量中。

核心指标：

Compare_PCC — YUV 域处理前后的相关性，应接近 1
Compare_Delta_SNR_dB — 如果为负，说明该模块在损害信号
SampleEntropy — 样本熵变化，如果处理后突增说明引入了不可预测的噪声
Peak_Prominence_Ratio — 峰值突出度，反映心跳峰是否清晰可辨
重点关注：

ColorSpaceConversion：纯数学变换，理论上无损。Compare_PCC 应 ≈ 1
ContrastSaturation：你当前 factor=1.0（无操作），应该完全透传。如果指标有任何变化说明实现有 bug
YUVtoRGB：同理，纯数学逆变换
