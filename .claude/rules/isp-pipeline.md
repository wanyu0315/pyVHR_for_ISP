# ISP 流水线规则

## 处理顺序

```
RAW → BLC → DPC → [RawDenoise] → WB → Demosaic → CCM → Gamma
    → RGB→YUV → [Denoise] → [Sharpen] → ContrastSat → YUV→RGB → Output
```

## 关键设计决策

- **编码格式**：FFV1 全 I 帧无损（GOP=1），禁止使用 H.264/H.265
- **像素格式**：`bgr0` / `bgr48le`，确保空间无损
- **参数记录**：每次运行自动保存完整 ISP 参数到 JSON，确保可复现
- **输出位深**：支持 8-bit 和 16-bit
