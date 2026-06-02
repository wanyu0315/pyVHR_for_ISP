# 消融实验规则

## 设计原则

- **单一变量控制**：每次只改变一个 ISP 参数，保持其他参数不变
- **多受试者验证**：每组实验至少覆盖 10+ 名受试者，避免个体差异干扰结论
- **结果可比性**：同一组实验使用相同的 pyVHR 参数（method、winsize、roi_method 等）
- **输出目录**：`sensitivity_analysis/<环境>/<ISP模块>/<参数值>/`

## 进度

已完成：
- [x] Gamma 消融（γ = 0.8, 1.0, 2.2, 3.0）
- [x] 基线 ISP 配置 + 11 名受试者
- [x] Bayer-aware 四通道采样（R/G1/G2/B）
- [x] RAW→RGB 跨域对比（RAW_G vs RGB_G）

待完成：
- [ ] Demosaic 算法消融（bilinear vs AHD vs malvar2004）
- [ ] 白平衡算法消融（gray_world_green vs fixed_gain）
- [ ] CCM 矩阵消融（sensor_to_srgb vs identity）
- [ ] RAW 域降噪消融（None vs gaussian vs bilateral）
- [ ] 组合消融：Gamma × Demosaic 交叉实验
- [ ] 探针 RGB 时序接入 pyVHR CHROM/POS 方法，直接在中间节点估计 BPM
- [ ] cpu_POS / cpu_LGI / cpu_ICA 方法对比测试