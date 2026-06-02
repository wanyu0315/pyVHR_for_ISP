# rPPG 分析规则


## 标准 Pipeline 调用方式

```python
from pyVHR.analysis.pipeline import Pipeline
pipe = Pipeline()
bvps, timesES, bpmES = pipe.run_on_video(
    videoFileName=video_path,
    winsize=16,                    # 时间窗口大小（秒）
    roi_method="convexhull",
    roi_approach="holistic",
    method="cpu_CHROM",            # rPPG 算法
    estimate="holistic",
    patch_size=40,
    RGB_LOW_HIGH_TH=(75, 230),
    Skin_LOW_HIGH_TH=(75, 230),
    pre_filt=True,
    post_filt=True,
    cuda=True,
    verb=True
)
```

## 误差评估标准接口

```python
from pyVHR.utils.errors import getErrors, printErrors
RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
```

## pyVHR 内部流程参考

1. 视频读取 → RGB uint8
2. MediaPipe FaceMesh → ConvexHull mask（排除眼睛和嘴巴）
3. 皮肤 ROI 内 RGB 均值提取（参见 `holistic_mean()`）
4. rPPG 算法在时间窗口内估计 BVP
5. Welch PSD → 心率频段（0.7–2.0 Hz）峰值 → BPM
6. 带通滤波、异常值剔除

## 传感器参数

| 参数       | 值         |
| ---------- | ---------- |
| 分辨率     | 1280 × 800 |
| 位深       | 16-bit     |
| Bayer 排列 | GRBG       |
| 帧率       | 30 fps     |
| 数据格式   | 无头 .raw  |