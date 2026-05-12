"""多算法 rPPG BPM 估计公共函数.

本模块是探针分析系统中 Layer 1 (算法依赖层) 和 proxy-GT 生成的公共底层.
给定 ROI 时序输入, 调用 pyVHR 的 rPPG 算法 (GREEN/CHROM/POS/LGI) 提取 BVP,
按滑窗 Welch 估计 BPM 序列, 输出供上层作两套用途:
  1. Layer 1 指标: 在每个探针位置给出多算法 BPM/SNR.
  2. proxy-GT 生成: 在 Input 探针位置把 BPM 序列导出为 .npz 供后续探针对比.

数值口径完全对齐 pyVHR (频段 0.65-4.0 Hz, Welch + nfft=2048, 窗口中心即时间),
避免探针端 bpmES 与 proxy-GT 端 bpmES 算法不一致导致的对比偏见.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# probe_ana_cmp.py / gt_reference.py 的常量定义在父目录 probe_analysis/ 下,
# 这里把父目录加入 sys.path 以便子模块导入共用常量与工具函数.
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_PARENT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_ana_cmp import (
    HR_BAND_HI,
    HR_BAND_LO,
    _pyvhr_welch,
    _pyvhr_window_ranges,
    build_probe_time_axis,
    preprocess,
    pyvhr_bpm_estimate,
    pyvhr_gt_snr_db,
)

from pyVHR.BVP.methods import cpu_CHROM, cpu_GREEN, cpu_LGI, cpu_POS


ALGO_GREEN = "GREEN"
ALGO_CHROM = "CHROM"
ALGO_POS = "POS"
ALGO_LGI = "LGI"
SUPPORTED_ALGOS: Tuple[str, ...] = (ALGO_GREEN, ALGO_CHROM, ALGO_POS, ALGO_LGI)

DOMAIN_RAW = "raw"
DOMAIN_RGB = "rgb"
DOMAIN_YUV = "yuv"
SUPPORTED_DOMAINS: Tuple[str, ...] = (DOMAIN_RAW, DOMAIN_RGB, DOMAIN_YUV)

DEFAULT_WINDOW_SIZE_SEC = 16.0
DEFAULT_STRIDE_SEC = 1.0

# BT.709 YUV (full-range) -> RGB 逆变换, 探针在 YUV 域能复用 CHROM/POS 时用.
_YUV2RGB_BT709 = np.array(
    [
        [1.0, 0.0, 1.5748],
        [1.0, -0.1873, -0.4681],
        [1.0, 1.8556, 0.0],
    ],
    dtype=np.float64,
)


class MultiAlgoBpmError(ValueError):
    """compute_multi_algo_bpm 输入不合法或算法不适用时抛出."""


def _normalize_algos(algos: Optional[List[str]], domain: str, signal_is_3ch: bool) -> List[str]:
    """按域 + 通道数过滤不适用算法, 返回合法大写算法列表.

    RAW 域单通道 → 仅 GREEN;
    RAW 域 Bayer-aware 三通道 (RAW_R/G/B) → 支持全部算法
      (Bayer 曝光值近似 sRGB 响应, rPPG 文献有此先例, 论文表述为 "raw Bayer 响应下的 CHROM/POS").
    RGB/YUV → 全部算法.
    """
    if algos is None:
        candidates = list(SUPPORTED_ALGOS)
    else:
        candidates = [str(a).upper() for a in algos]
    unknown = [a for a in candidates if a not in SUPPORTED_ALGOS]
    if unknown:
        raise MultiAlgoBpmError(f"Unsupported algos: {unknown}")
    if domain == DOMAIN_RAW and not signal_is_3ch:
        filtered = [a for a in candidates if a == ALGO_GREEN]
    else:
        filtered = candidates
    # 按 SUPPORTED_ALGOS 顺序去重返回, 避免外部乱序.
    ordered = [a for a in SUPPORTED_ALGOS if a in filtered]
    if not ordered:
        raise MultiAlgoBpmError(
            f"No applicable algorithm after domain filtering "
            f"(domain={domain}, signal_is_3ch={signal_is_3ch}, requested={algos})."
        )
    return ordered


def _ensure_rgb_matrix(signal: np.ndarray, domain: str) -> np.ndarray:
    """将输入信号统一成 [N, 3] RGB 浮点矩阵.

    - RGB 域: 直接检查形状.
    - RAW 域 单通道: 仅 GREEN 可用; 内部把单通道复制成三列占位.
    - RAW 域 三通道 (Bayer-aware): 直接当作 RGB 使用 (不做任何变换).
    - YUV 域: 通过 BT.709 逆变换回 RGB.
    """
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        if domain != DOMAIN_RAW:
            raise MultiAlgoBpmError(
                f"Domain {domain} expects 3-channel signal [N, 3], got 1D."
            )
        n = arr.shape[0]
        # RAW 单通道: 用 G 列放真实信号, R/B 填 0. 仅 GREEN 会用到.
        rgb = np.zeros((n, 3), dtype=np.float64)
        rgb[:, 1] = arr
        return rgb
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise MultiAlgoBpmError(
            f"RGB/RAW-3ch/YUV signal must be shape [N, 3], got {arr.shape}."
        )
    if domain == DOMAIN_YUV:
        return arr @ _YUV2RGB_BT709.T
    # RGB 与 RAW 三通道统一视为 [N, 3] 直通.
    return arr


def _preprocess_channels(rgb: np.ndarray) -> np.ndarray:
    """逐通道做与探针主管道一致的 preprocess (中值 + detrend 保留 DC)."""
    out = np.empty_like(rgb)
    for ch in range(rgb.shape[1]):
        out[:, ch] = preprocess(rgb[:, ch])
    return out


def _run_rppg_algo(algo: str, rgb_window: np.ndarray, fps: float) -> np.ndarray:
    """对一个 [N, 3] 窗口执行算法, 返回 1D BVP (长度 N).

    pyVHR 期望 [num_estimators, 3, num_frames], 我们单 ROI 对应 estimators=1.
    """
    x = np.asarray(rgb_window, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3:
        raise MultiAlgoBpmError(
            f"_run_rppg_algo expects [N, 3], got {x.shape}."
        )
    stacked = np.transpose(x, (1, 0))[np.newaxis, :, :]  # [1, 3, N]
    if algo == ALGO_GREEN:
        bvp = cpu_GREEN(stacked)
    elif algo == ALGO_CHROM:
        bvp = cpu_CHROM(stacked)
    elif algo == ALGO_POS:
        bvp = cpu_POS(stacked, fps=float(fps))
    elif algo == ALGO_LGI:
        bvp = cpu_LGI(stacked)
    else:
        raise MultiAlgoBpmError(f"Unsupported algo: {algo}")
    bvp_arr = np.asarray(bvp, dtype=np.float64)
    if bvp_arr.ndim != 2 or bvp_arr.shape[0] != 1:
        raise MultiAlgoBpmError(
            f"Algo {algo} returned unexpected shape {bvp_arr.shape}, expected [1, N]."
        )
    return bvp_arr[0]


def compute_multi_algo_bpm(
    signal: np.ndarray,
    frame_ids: np.ndarray,
    fps: float,
    algos: Optional[List[str]] = None,
    domain: str = DOMAIN_RGB,
    window_size_sec: float = DEFAULT_WINDOW_SIZE_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    probe_time_mode: str = "absolute_frame_id",
    probe_time_offset_sec: float = 0.0,
    do_preprocess: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """对 ROI 时序做多 rPPG 算法 + 滑窗 BPM 估计.

    Args:
        signal: RGB/YUV 域 shape=[N, 3]; RAW 域 shape=[N].
        frame_ids: 与 signal 长度相同的帧号数组.
        fps: 采样率.
        algos: 要运行的算法列表, None 表示全部. RAW 域会自动过滤到仅 GREEN.
        domain: 'rgb' / 'raw' / 'yuv'.
        window_size_sec / stride_sec: 滑窗参数, 与 pyVHR 16s/1s 默认一致.
        probe_time_mode / probe_time_offset_sec: 时间轴构建参数.
        do_preprocess: 是否跑 probe_ana_cmp.preprocess. proxy-GT 场景应开启.

    Returns:
        dict 形如:
        {
          'GREEN': {
              'bpm':        np.ndarray (N_win,),
              'times':      np.ndarray (N_win,),
              'bvp_full':   np.ndarray (N,)   # 整段 BVP, 便于外层加工
              'bvp_windows':list[np.ndarray]  # 每个窗口的 BVP 片段
              'ranges':     list[(start, end)]
          },
          ...
        }
    """
    if domain not in SUPPORTED_DOMAINS:
        raise MultiAlgoBpmError(f"Unsupported domain: {domain}")
    if fps <= 0:
        raise MultiAlgoBpmError(f"fps must be positive, got {fps}.")

    raw_arr = np.asarray(signal)
    signal_is_3ch = raw_arr.ndim == 2 and raw_arr.shape[-1] == 3
    resolved_algos = _normalize_algos(algos, domain, signal_is_3ch)

    rgb = _ensure_rgb_matrix(signal, domain)
    if rgb.shape[0] != len(frame_ids):
        raise MultiAlgoBpmError(
            f"signal length ({rgb.shape[0]}) and frame_ids length "
            f"({len(frame_ids)}) mismatch."
        )
    if do_preprocess:
        if domain == DOMAIN_RAW and not signal_is_3ch:
            # RAW 单通道只在 G 列有效, 只对 G 列做 preprocess.
            rgb = rgb.copy()
            rgb[:, 1] = preprocess(rgb[:, 1])
        else:
            rgb = _preprocess_channels(rgb)

    time_axis = build_probe_time_axis(
        np.asarray(frame_ids),
        fps,
        mode=probe_time_mode,
        offset_sec=probe_time_offset_sec,
    )
    ranges = _pyvhr_window_ranges(rgb.shape[0], fps, window_size_sec, stride_sec)
    if not ranges:
        # 信号太短, 为每个算法返回空结果, 上层负责容错.
        empty = {
            "bpm": np.array([], dtype=float),
            "times": np.array([], dtype=float),
            "bvp_full": np.zeros(rgb.shape[0], dtype=float),
            "bvp_windows": [],
            "ranges": [],
        }
        return {algo: dict(empty, bvp_full=empty["bvp_full"].copy()) for algo in resolved_algos}

    centers = np.array(
        [0.5 * (time_axis[start] + time_axis[end - 1]) for start, end in ranges],
        dtype=float,
    )

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for algo in resolved_algos:
        # 1) 计算整段 BVP, 便于外层导出 / 可视化.
        try:
            bvp_full = _run_rppg_algo(algo, rgb, fps)
        except Exception as exc:  # pragma: no cover - 保底防御
            raise MultiAlgoBpmError(
                f"Failed to compute full BVP for algo {algo} on domain {domain}: {exc}"
            ) from exc

        # 2) 再逐窗口计算 BVP, 保证窗口边界与 BPM 估计严格一致.
        bvp_windows: List[np.ndarray] = []
        bpm_list: List[float] = []
        for start, end in ranges:
            window_rgb = rgb[start:end]
            bvp_window = _run_rppg_algo(algo, window_rgb, fps)
            bvp_windows.append(bvp_window)
            bpm_list.append(pyvhr_bpm_estimate(bvp_window, fps))

        results[algo] = {
            "bpm": np.asarray(bpm_list, dtype=float),
            "times": centers.copy(),
            "bvp_full": bvp_full,
            "bvp_windows": bvp_windows,
            "ranges": list(ranges),
        }
    return results


def compute_multi_algo_gt_snr(
    algo_result: Dict[str, np.ndarray],
    fps: float,
    gt_bpm_per_window: np.ndarray,
) -> np.ndarray:
    """对某算法每个窗口算 pyVHR 风格 GT-anchored SNR, 返回 (N_win,) dB 数组.

    外层若已拿到 GT BPM (来自 PPG GT 或 proxy-GT), 就可不经 gt_metrics_for_signal,
    直接复用这里算出 SNR. 主要用于 Layer 1 的诊断字段.
    """
    windows = algo_result.get("bvp_windows", [])
    gt = np.asarray(gt_bpm_per_window, dtype=float)
    if len(windows) != gt.shape[0]:
        raise MultiAlgoBpmError(
            f"Window count mismatch: windows={len(windows)} vs gt={gt.shape[0]}."
        )
    out = np.full(len(windows), np.nan, dtype=float)
    for i, window in enumerate(windows):
        if not np.isfinite(gt[i]) or gt[i] <= 0:
            continue
        out[i] = pyvhr_gt_snr_db(np.asarray(window, dtype=float), fps, float(gt[i]))
    return out


def load_reference_rgb_from_csv(
    reference_csv: str,
    channel_prefix: str = "ROI_Mean",
    prefer_raw_bayer_rgb: bool = True,
    fallback_raw_green: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """从 reference CSV 读取信号与 Frame_ID, 返回 (signal, frame_ids, domain).

    优先级 (针对 raw_proxyGT_data/raw_proxyGT_reference 下的 CSV,
             其 ROI_Mean_C0/C1/C2 中 C1/C2 为 N/A):
      1. prefer_raw_bayer_rgb=True 且 RAW_R/G/B_Mean 三列都有效
         → ([N, 3] Bayer-aware 三通道, 'raw')  支持全算法
      2. ROI_Mean_C0/C1/C2 三列都有效
         → ([N, 3] RGB, 'rgb')  支持全算法
      3. fallback_raw_green=True 且 RAW_G_Mean 有效
         → ([N] 单通道, 'raw')  仅 GREEN
      4. ROI_Mean_C0 有效
         → ([N] 单通道, 'raw')  仅 GREEN

    否则抛 MultiAlgoBpmError.
    """
    if not os.path.isfile(reference_csv):
        raise MultiAlgoBpmError(f"Reference CSV not found: {reference_csv}")
    df = pd.read_csv(reference_csv)
    if "Frame_ID" not in df.columns:
        raise MultiAlgoBpmError("Reference CSV missing Frame_ID column.")
    frame_ids = pd.to_numeric(df["Frame_ID"], errors="coerce").to_numpy()

    def _valid_numeric(col: str) -> Optional[np.ndarray]:
        if col not in df.columns:
            return None
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        return vals if np.isfinite(vals).sum() > 10 else None

    # 路径 1: Bayer-aware 三通道 (raw_proxyGT_data 的默认情况)
    if prefer_raw_bayer_rgb:
        r = _valid_numeric("RAW_R_Mean")
        g = _valid_numeric("RAW_G_Mean")
        b = _valid_numeric("RAW_B_Mean")
        if r is not None and g is not None and b is not None:
            rgb = np.stack([np.nan_to_num(r), np.nan_to_num(g), np.nan_to_num(b)], axis=1)
            return rgb, frame_ids, DOMAIN_RAW

    # 路径 2: 标准 RGB 三通道 (ROI_Mean_C0/C1/C2)
    col_c0 = f"{channel_prefix}_C0"
    col_c1 = f"{channel_prefix}_C1"
    col_c2 = f"{channel_prefix}_C2"
    c0 = _valid_numeric(col_c0)
    c1 = _valid_numeric(col_c1)
    c2 = _valid_numeric(col_c2)
    if c0 is not None and c1 is not None and c2 is not None:
        rgb = np.stack([np.nan_to_num(c0), np.nan_to_num(c1), np.nan_to_num(c2)], axis=1)
        return rgb, frame_ids, DOMAIN_RGB

    # 路径 3: RAW 单通道 fallback
    if fallback_raw_green:
        g = _valid_numeric("RAW_G_Mean")
        if g is not None:
            return np.nan_to_num(g), frame_ids, DOMAIN_RAW
        if c0 is not None:
            return np.nan_to_num(c0), frame_ids, DOMAIN_RAW

    raise MultiAlgoBpmError(
        f"Reference CSV {reference_csv} has no usable RGB or RAW Bayer channels."
    )
