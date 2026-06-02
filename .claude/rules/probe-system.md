# 探针系统规则

## 设计原则

- 探针（PipelineProbe）是**纯观测模块**，不修改图像数据，只采集统计量
- 同一物理帧内所有探针共享 `ProbeSkinContext`，只做一次 landmarks + skin mask 检测
- RAW 域使用 Bayer-aware 四通道采样（R/G1/G2/B），必须在原图坐标系进行 CFA 分类，不能先裁 ROI 再分类

## 探针系统的 rPPG 分析规则

### 核心原则：必须使用 pyVHR 标准实现

- **rPPG 信号提取**：必须使用 `pyVHR.analysis.pipeline.Pipeline`，禁止自行实现 CHROM/POS/ICA
- **皮肤检测与 ROI 提取**：使用 pyVHR 的 `SkinExtractionConvexHull` 或 MediaPipe FaceMesh
- **误差评估**：使用 `pyVHR.utils.errors.getErrors()`，禁止自行计算 RMSE/MAE/PCC/CCC/SNR
- **信号提取方法**：参照 `sig_extraction.py` 中的 `holistic_mean()` 和 `landmarks_mean()`
- **皮肤亮度过滤**：`RGB_LOW_TH=55 / RGB_HIGH_TH=230`

仅当 pyVHR 不提供所需功能时（如 ISP 探针 RAW 域分析），才允许自行实现，且须对齐 pyVHR 方法论。

## 域判定与通道选择

| 域    | 主分析通道            |
| ----- | --------------------- |
| RAW   | `RAW_G_Mean`          |
| RGB   | `ROI_Mean_C1(G)`      |
| YUV   | `ROI_Mean_C0(Y)`      |

## 分析维度

- **时域**：SNR_dB、AC_PeakToPeak、AC/DC 比率
- **频域**：HR_Band_Ratio、BPM 估计、频谱熵、谐波比
- **HRV**：SDNN、RMSSD、pNN50、LF/HF 比
- **信号质量**：自相关峰值、样本熵、峰值突出度
- **跨探针对比**：PCC、相位延迟、BPM 偏移、频谱余弦相似度、ΔSNR

## 元数据规范

- 探针元数据（domain、bayer_pattern、roi_mode 等）写入 `probe_meta.json`
- Ground Truth 使用 `.npz` 格式（`bpm` + `times` + `metadata`）