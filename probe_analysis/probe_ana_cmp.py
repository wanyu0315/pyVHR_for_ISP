"""
ISP 探针 CSV 时序数据的生理信号对比分析 (域感知版本)
支持 RAW / RGB / YUV 三种域的自动检测与通道选择
用法: python probe_comparison.py <probe1_csv> <probe2_csv> [--fps 30] [--output results/] [--domain auto]
文件结构:
probe_analysis/
├── probe_comparison.py          # 主分析脚本
└── results/
    ├── metrics_summary.csv      # 指标汇总
    ├── timeseries_comparison.png
    ├── psd_comparison.png
    ├── spectrogram.png
    └── hrv_poincare.png
"""
import argparse
from itertools import combinations
import json
import os
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

# gt_reference 已移至 probe_analysis/GTPreprocessing/ 子目录, 以下按本模块所在目录
# 计算子目录路径并加入 sys.path, 保证 "python probe_ana_cmp.py" 和 "python -m"
# 两种调用方式都能找到 gt_reference.
import sys as _sys
_PROBE_ANA_DIR = os.path.dirname(os.path.abspath(__file__))
_GT_PREP_DIR = os.path.join(_PROBE_ANA_DIR, "GTPreprocessing")
for _p in (_PROBE_ANA_DIR, _GT_PREP_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    from gt_reference import CONTACT_PPG_SOURCE, DEFAULT_GT_ROOT, load_gt_reference
except ImportError:  # pragma: no cover - supports package-style execution
    from .GTPreprocessing.gt_reference import (
        CONTACT_PPG_SOURCE,
        DEFAULT_GT_ROOT,
        load_gt_reference,
    )
_cjk_ttf = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
_zh_prop = None
if os.path.isfile(_cjk_ttf):
    fm.fontManager.addfont(_cjk_ttf)
    _zh_prop = fm.FontProperties(fname=_cjk_ttf)
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _apply_zh(artist) -> None:
    """对包含中文的 matplotlib Text 对象显式应用 WenQuanYi 字体（rcParams 失效时的回退）。"""
    if _zh_prop is not None and artist is not None:
        try:
            artist.set_fontproperties(_zh_prop)
        except Exception:
            pass

# NumPy 2.0 兼容：trapezoid 是 trapz 的新名（trapz 在 NumPy 2.0 已 deprecated）
_trapz = getattr(np, "trapezoid", None) or np.trapz

HR_BAND_LO = 0.65
HR_BAND_HI = 4.0
SNR_PEAK_HALF_WIDTH = 0.2
SNR_NOISE_LO = HR_BAND_LO
SNR_NOISE_HI = HR_BAND_HI
PYVHR_BPM_NFFT = 2048
PYVHR_SNR_FRES_BPM = 0.5
PYVHR_SNR_HALF_WIDTH_BPM = 0.2 * 60.0
GT_DEFAULT_WINDOW_SIZE = 16.0
GT_DEFAULT_STRIDE = 1.0
GT_WINDOW_METRIC_COLUMNS = [
    'Probe',
    'Window_Index',
    'Window_Start_Frame',
    'Window_End_Frame',
    'Window_Center_s',
    'GT_BPM_For_Error',
    'GT_BPM_For_SNR',
    'BPM_Estimate',
    'BPM_Error_EstMinusGT',
    'GT_SNR_dB',
]

RAW_MULTI_CHANNELS = [
    ('RAW_R', 'RAW_R_Mean', 'RAW_R_AC_Delta', '#d62728'),
    ('RAW_G1', 'RAW_G1_Mean', 'RAW_G1_AC_Delta', '#ff7f0e'),
    ('RAW_G2', 'RAW_G2_Mean', 'RAW_G2_AC_Delta', '#17becf'),
    ('RAW_G', 'RAW_G_Mean', 'RAW_G_AC_Delta', '#9467bd'),
    ('RAW_B', 'RAW_B_Mean', 'RAW_B_AC_Delta', '#1f77b4'),
]

RAW_MULTI_KEYS = [item[0] for item in RAW_MULTI_CHANNELS]

# ============================================================
# 模块 0: 数据加载与预处理
# ============================================================

def load_probe_meta(csv_path: str) -> dict:
    """读取与 CSV 同目录的 probe 元数据，缺失时返回空 dict。"""
    meta_path = os.path.join(os.path.dirname(csv_path), 'probe_meta.json')
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def detect_domain(csv_path: str, df: pd.DataFrame, meta: dict = None) -> str:
    meta = meta or {}
    name = os.path.basename(os.path.dirname(csv_path)).lower()
    if meta.get('raw_mode') == 'bayer_aware' or meta.get('raw_bayer_pattern'):
        return 'raw'
    # 特殊处理：YUVtoRGB-Output 是 RGB 域（转换后的输出）
    if name == 'output' or name.endswith('-output'):
        return 'rgb'
    # YUV域：前一模块是 colorspace/contrastsaturation
    yuv_prefixes = ['colorspace-', 'contrastsaturation-']
    if any(name.startswith(k) for k in yuv_prefixes):
        return 'yuv'
    # RAW域：C1 全为 NaN
    c1 = pd.to_numeric(df.get('ROI_Mean_C1', pd.Series(dtype=float)), errors='coerce')
    if c1.isna().all():
        return 'raw'
    return 'rgb'


def load_and_align(csv1: str, csv2: str):
    """读取两个探针 CSV，按 Frame_ID 对齐，插值 N/A"""
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    for df in (df1, df2):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df1 = df1.rename(columns={col: f'{col}_p1' for col in df1.columns if col != 'Frame_ID'})
    df2 = df2.rename(columns={col: f'{col}_p2' for col in df2.columns if col != 'Frame_ID'})
    merged = pd.merge(df1, df2, on='Frame_ID')
    merged = merged.sort_values('Frame_ID').reset_index(drop=True)
    merged = merged.interpolate(method='linear', limit_direction='both')
    merged = merged.dropna(subset=['Frame_ID'])
    return merged


def bandpass_filter(x: np.ndarray, fps: float, lo: float = HR_BAND_LO, hi: float = HR_BAND_HI, order: int = 4):
    """Butterworth 带通滤波（含上下频段边界保护）"""
    if len(x) < 30:
        return np.zeros_like(x)
    nyq = fps / 2.0
    if lo <= 0:
        lo = 0.01
    if hi >= nyq:
        hi = nyq - 0.01
    if lo >= hi:
        return np.zeros_like(x)
    b, a = sig.butter(order, [lo / nyq, hi / nyq], btype='band')
    return sig.filtfilt(b, a, x)


def remove_dc(x: np.ndarray):
    return x - np.mean(x)


def preprocess(x: np.ndarray):
    """预处理：中值滤波去尖刺 + 线性去趋势，保留原始 DC 偏置

    注意：必须在 detrend 前保存 DC，detrend 后信号均值≈0，
    若再用 np.mean(x) 加回则等同于不加（旧代码 bug）。
    """
    dc = float(np.mean(x))
    x = median_filter(x, size=3)
    x = sig.detrend(x, type='linear')
    return x + dc

# ============================================================
# 公共 SNR 计算（全局统一公式）
# ============================================================
# 时域 SNR：对原始信号做 PSD，衡量心率主频在频域中的突出程度。
#           值越高说明原始信号中心率成分越纯净，噪声/运动伪影越少。
# 频域 SNR：对 PSD 曲线上直接计算，含义与时域相同，但上下文不同：
#           PSD 图中直接可视化了信号功率与噪声底噪的对比。
# 统一公式：
#   f_peak = argmax(PSD) in [0.65, 4.0] Hz
#   Signal = ∫ PSD(f) df,  f ∈ [max(0.65, f_peak - 0.2), min(4.0, f_peak + 0.2)]
#   Noise  = ∫ PSD(f) df,  f ∈ [0.65, 4.0] \ Signal
#   SNR_dB = 10 * log10(Signal / Noise)

def peak_snr_db(x: np.ndarray, fps: float) -> float:
    """统一的峰值 SNR 计算：0.65-4.0Hz 主频 ±0.2Hz 能量 vs 0.65-4.0Hz 剩余能量"""
    nperseg = min(256, max(16, len(x) // 4))
    freqs, psd = sig.welch(remove_dc(x), fs=fps, nperseg=nperseg)

    hr_mask = (freqs >= HR_BAND_LO) & (freqs <= HR_BAND_HI)
    if not np.any(hr_mask) or np.max(psd[hr_mask]) <= 0:
        return 0.0

    peak_idx = np.argmax(psd[hr_mask])
    peak_freq = freqs[hr_mask][peak_idx]

    peak_lo = max(SNR_NOISE_LO, peak_freq - SNR_PEAK_HALF_WIDTH)
    peak_hi = min(SNR_NOISE_HI, peak_freq + SNR_PEAK_HALF_WIDTH)
    peak_band = (freqs >= peak_lo) & (freqs <= peak_hi)
    signal_power = _trapz(psd[peak_band], freqs[peak_band])

    noise_mask = ~peak_band & (freqs >= SNR_NOISE_LO) & (freqs <= SNR_NOISE_HI)
    noise_power = _trapz(psd[noise_mask], freqs[noise_mask])

    if signal_power <= 0 or noise_power <= 0:
        return 0.0
    return 10 * np.log10(signal_power / noise_power)


# ============================================================
# pyVHR 兼容 GT 指标
# ============================================================

def build_probe_time_axis(frame_ids: np.ndarray, fps: float, mode: str = 'absolute_frame_id',
                          offset_sec: float = 0.0) -> np.ndarray:
    """Build probe timestamps in seconds for GT alignment."""
    frames = np.asarray(frame_ids, dtype=float)
    if frames.size == 0:
        return np.array([], dtype=float)
    if fps <= 0:
        raise ValueError("fps must be positive when building probe time axis.")
    if mode == 'absolute_frame_id':
        return frames / fps + float(offset_sec)
    if mode == 'relative_csv':
        return (frames - frames[0]) / fps + float(offset_sec)
    raise ValueError(f"Unsupported probe_time_mode: {mode}")


def _pyvhr_snr_nfft(fps: float) -> int:
    nyquist = fps / 2.0
    return int(np.ceil((60.0 * 2.0 * nyquist) / PYVHR_SNR_FRES_BPM))


def _pyvhr_welch(bvps: np.ndarray, fps: float, nfft: int = PYVHR_BPM_NFFT,
                 min_hz: float = HR_BAND_LO, max_hz: float = HR_BAND_HI):
    """Local implementation of pyVHR.BPM.utils.Welch without importing pyVHR."""
    data = np.asarray(bvps, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] < 2:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    _, n = data.shape
    if n < 256:
        seglength = n
        overlap = int(0.8 * n)
    else:
        seglength = 256
        overlap = 200
    overlap = min(overlap, max(0, seglength - 1))
    nfft = max(int(nfft), seglength)

    freqs, power = sig.welch(
        data,
        nperseg=seglength,
        noverlap=overlap,
        fs=fps,
        nfft=nfft,
    )
    band = np.argwhere((freqs > min_hz) & (freqs < max_hz)).flatten()
    return (60.0 * freqs[band]).astype(np.float32), power[:, band].astype(np.float32)


def pyvhr_bpm_estimate(window: np.ndarray, fps: float) -> float:
    """pyVHR BPM.BVP_to_BPM-compatible PSD peak BPM estimate."""
    pfreqs, power = _pyvhr_welch(window, fps, nfft=PYVHR_BPM_NFFT)
    if power.size == 0 or pfreqs.size == 0:
        return np.nan
    p = power[0]
    if not np.any(np.isfinite(p)) or np.nanmax(p) <= 0:
        return np.nan
    return float(pfreqs[int(np.nanargmax(p))])


def pyvhr_gt_snr_db(window: np.ndarray, fps: float, gt_bpm: float) -> float:
    """pyVHR get_SNR-compatible GT-referenced SNR for one BVP window."""
    if not np.isfinite(gt_bpm) or gt_bpm <= 0:
        return np.nan
    pfreqs, power = _pyvhr_welch(window, fps, nfft=_pyvhr_snr_nfft(fps))
    if power.size == 0 or pfreqs.size == 0:
        return np.nan

    mask_hr = (pfreqs >= gt_bpm - PYVHR_SNR_HALF_WIDTH_BPM) & (
        pfreqs <= gt_bpm + PYVHR_SNR_HALF_WIDTH_BPM
    )
    mask_h1 = (pfreqs >= gt_bpm * 2.0 - PYVHR_SNR_HALF_WIDTH_BPM) & (
        pfreqs <= gt_bpm * 2.0 + PYVHR_SNR_HALF_WIDTH_BPM
    )
    signal_mask = mask_hr | mask_h1
    noise_mask = ~signal_mask

    snrs = []
    for row in power:
        p = np.nan_to_num(row.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        signal_power = np.sum(p[signal_mask])
        noise_power = np.sum(p[noise_mask])
        if signal_power <= 0 or noise_power <= 0:
            snrs.append(np.nan)
        else:
            snrs.append(10.0 * np.log10(signal_power / noise_power))
    snrs = np.asarray(snrs, dtype=float)
    if not np.any(np.isfinite(snrs)):
        return np.nan
    return float(np.nanmedian(snrs))


def _pyvhr_window_ranges(n_samples: int, fps: float, window_size_sec: float, stride_sec: float):
    wsize_fr = int(round(float(window_size_sec) * fps))
    stride_fr = int(round(float(stride_sec) * fps))
    if wsize_fr <= 1 or stride_fr <= 0 or n_samples < wsize_fr:
        return []
    num_win = int((n_samples - wsize_fr) / stride_fr) + 1
    ranges = []
    for idx in range(num_win):
        start = idx * stride_fr
        end = start + wsize_fr
        if end <= n_samples:
            ranges.append((start, end))
    return ranges


def _concordance_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    cor = np.corrcoef(x, y)[0][1]
    numerator = 2.0 * cor * np.std(x) * np.std(y)
    denominator = np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2
    return float(numerator / denominator) if denominator > 0 else np.nan


def gt_metrics_for_signal(signal: np.ndarray, frame_ids: np.ndarray, fps: float, gt_ref,
                          window_size_sec: float = GT_DEFAULT_WINDOW_SIZE,
                          stride_sec: float = GT_DEFAULT_STRIDE,
                          probe_time_mode: str = 'absolute_frame_id',
                          probe_time_offset_sec: float = 0.0,
                          probe_name: str = ''):
    """Compute pyVHR-compatible GT metrics for one probe signal."""
    metrics = {
        'GT_Window_Size_s': float(window_size_sec),
        'GT_Stride_s': float(stride_sec),
        'GT_Probe_Time_Mode': probe_time_mode,
        'GT_Probe_Time_Offset_s': float(probe_time_offset_sec),
        'Valid_GT_Window_Count': 0,
        'Valid_GT_SNR_Window_Count': 0,
        'GT_SNR_dB': np.nan,
        'GT_SNR_Median_dB': np.nan,
        'GT_SNR_Std_dB': np.nan,
        'BPM_MAE': np.nan,
        'BPM_RMSE': np.nan,
        'BPM_MAXError': np.nan,
        'BPM_Bias': np.nan,
        'BPM_PCC': np.nan,
        'BPM_CCC': np.nan,
        'GT_BPM_Mean': np.nan,
        'GT_BPM_Min': np.nan,
        'GT_BPM_Max': np.nan,
        'GT_First_Window_Center_s': np.nan,
        'GT_Last_Window_Center_s': np.nan,
    }
    window_rows = []

    time_axis = build_probe_time_axis(frame_ids, fps, probe_time_mode, probe_time_offset_sec)
    ranges = _pyvhr_window_ranges(len(signal), fps, window_size_sec, stride_sec)
    if not ranges:
        return metrics, window_rows

    centers = np.asarray([0.5 * (time_axis[start] + time_axis[end - 1]) for start, end in ranges], dtype=float)
    gt_bpm_for_error = gt_ref.bpm_at(centers, method='nearest')
    gt_bpm_for_snr = gt_ref.bpm_at(centers, method='pyvhr_index')
    metrics['GT_First_Window_Center_s'] = float(centers[0])
    metrics['GT_Last_Window_Center_s'] = float(centers[-1])

    bpm_estimates = []
    snrs = []
    for idx, (start, end) in enumerate(ranges):
        window = np.asarray(signal[start:end], dtype=float)
        bpm_est = pyvhr_bpm_estimate(window, fps)
        snr_db = pyvhr_gt_snr_db(window, fps, gt_bpm_for_snr[idx])
        bpm_estimates.append(bpm_est)
        snrs.append(snr_db)
        gt_err = gt_bpm_for_error[idx]
        bpm_error = bpm_est - gt_err if np.isfinite(bpm_est) and np.isfinite(gt_err) else np.nan
        window_rows.append({
            'Probe': _display_probe_name(probe_name) if probe_name else '',
            'Window_Index': idx,
            'Window_Start_Frame': int(frame_ids[start]),
            'Window_End_Frame': int(frame_ids[end - 1]),
            'Window_Center_s': float(centers[idx]),
            'GT_BPM_For_Error': float(gt_err) if np.isfinite(gt_err) else np.nan,
            'GT_BPM_For_SNR': float(gt_bpm_for_snr[idx]) if np.isfinite(gt_bpm_for_snr[idx]) else np.nan,
            'BPM_Estimate': float(bpm_est) if np.isfinite(bpm_est) else np.nan,
            'BPM_Error_EstMinusGT': float(bpm_error) if np.isfinite(bpm_error) else np.nan,
            'GT_SNR_dB': float(snr_db) if np.isfinite(snr_db) else np.nan,
        })

    bpm_estimates = np.asarray(bpm_estimates, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    valid_snr = np.isfinite(snrs)
    if np.any(valid_snr):
        metrics['Valid_GT_SNR_Window_Count'] = int(np.sum(valid_snr))
        metrics['GT_SNR_dB'] = float(np.mean(snrs[valid_snr]))
        metrics['GT_SNR_Median_dB'] = float(np.median(snrs[valid_snr]))
        metrics['GT_SNR_Std_dB'] = float(np.std(snrs[valid_snr]))

    valid_bpm = np.isfinite(bpm_estimates) & np.isfinite(gt_bpm_for_error)
    if np.any(valid_bpm):
        est = bpm_estimates[valid_bpm]
        gt = gt_bpm_for_error[valid_bpm]
        diff = est - gt
        abs_diff = np.abs(diff)
        metrics['Valid_GT_Window_Count'] = int(np.sum(valid_bpm))
        metrics['BPM_MAE'] = float(np.mean(abs_diff))
        metrics['BPM_RMSE'] = float(np.sqrt(np.mean(diff ** 2)))
        metrics['BPM_MAXError'] = float(np.max(abs_diff))
        metrics['BPM_Bias'] = float(np.mean(diff))
        metrics['GT_BPM_Mean'] = float(np.mean(gt))
        metrics['GT_BPM_Min'] = float(np.min(gt))
        metrics['GT_BPM_Max'] = float(np.max(gt))
        if len(est) >= 2 and np.std(est) > 1e-12 and np.std(gt) > 1e-12:
            metrics['BPM_PCC'] = float(pearsonr(gt, est)[0])
            metrics['BPM_CCC'] = _concordance_corrcoef(gt, est)

    return metrics, window_rows


def _store_gt_delta_metrics(all_metrics: dict, name1: str, name2: str):
    for metric in [
        'GT_SNR_dB',
        'GT_SNR_Median_dB',
        'BPM_MAE',
        'BPM_RMSE',
        'BPM_MAXError',
        'BPM_Bias',
        'BPM_PCC',
        'BPM_CCC',
    ]:
        v1 = all_metrics.get(f'{name1}_{metric}')
        v2 = all_metrics.get(f'{name2}_{metric}')
        if _is_number(v1) and _is_number(v2) and np.isfinite(v1) and np.isfinite(v2):
            all_metrics[f'Compare_Delta_{metric}'] = float(v2) - float(v1)
        else:
            all_metrics[f'Compare_Delta_{metric}'] = np.nan

# ============================================================
# 模块 1: 时域分析
# ============================================================

def time_domain_metrics(raw: np.ndarray, ac: np.ndarray, ac_delta: np.ndarray, fps: float):
    snr = peak_snr_db(raw, fps)

    return {
        'Mean': np.mean(raw),
        'Std': np.std(raw),
        'AC_PeakToPeak': np.ptp(ac),
        'SNR_dB': snr,
        'AC_Delta_Mean': np.mean(ac_delta),
        'AC_Delta_Std': np.std(ac_delta),
    }

# ============================================================
# 模块 2: 频域分析
# ============================================================

def freq_domain_metrics(raw: np.ndarray, fps: float):
    nperseg = min(256, max(16, len(raw) // 4))
    freqs, psd = sig.welch(remove_dc(raw), fs=fps, nperseg=nperseg)

    hr_mask = (freqs >= HR_BAND_LO) & (freqs <= HR_BAND_HI)
    hr_power = _trapz(psd[hr_mask], freqs[hr_mask])
    total_power = _trapz(psd, freqs)
    hr_ratio = hr_power / total_power if total_power > 0 else 0

    hr_freqs = freqs[hr_mask]
    hr_psd = psd[hr_mask]
    peak_freq = hr_freqs[np.argmax(hr_psd)] if len(hr_psd) > 0 else 0
    bpm = peak_freq * 60.0

    harmonic_ratio = 0.0
    if peak_freq > 0:
        h2_freq = peak_freq * 2
        h2_idx = np.argmin(np.abs(freqs - h2_freq))
        harmonic_ratio = psd[hr_mask][np.argmax(hr_psd)] / (psd[h2_idx] + 1e-12)

    psd_norm = psd[hr_mask] / (np.sum(psd[hr_mask]) + 1e-12)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))

    return {
        'HR_Band_Ratio': hr_ratio,
        'BPM_Estimate': bpm,
        'Peak_Freq_Hz': peak_freq,
        'Harmonic_Ratio': harmonic_ratio,
        'Spectral_Entropy': spectral_entropy,
    }, freqs, psd

# ============================================================
# 模块 3: HRV 分析
# ============================================================

def extract_rr_intervals(raw: np.ndarray, fps: float):
    bp = bandpass_filter(raw, fps)
    min_dist = max(1, int(fps / HR_BAND_HI))
    peaks, props = sig.find_peaks(bp, distance=min_dist, prominence=np.std(bp) * 0.3)
    if len(peaks) < 3:
        return np.array([]), peaks
    rr = np.diff(peaks) / fps * 1000.0
    return rr, peaks


def hrv_metrics(rr: np.ndarray, fps: float):
    if len(rr) < 3:
        return {k: np.nan for k in ['SDNN', 'RMSSD', 'pNN50', 'LF_HF_Ratio']}

    sdnn = np.std(rr, ddof=1)
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100.0

    rr_fs = 4.0
    rr_times = np.cumsum(rr) / 1000.0
    rr_times -= rr_times[0]
    t_uniform = np.arange(0, rr_times[-1], 1.0 / rr_fs)
    if len(t_uniform) < 8:
        return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50, 'LF_HF_Ratio': np.nan}
    rr_interp = np.interp(t_uniform, rr_times, rr)
    nperseg = min(64, len(rr_interp) // 2)
    if nperseg < 4:
        return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50, 'LF_HF_Ratio': np.nan}
    f_rr, psd_rr = sig.welch(rr_interp - np.mean(rr_interp), fs=rr_fs, nperseg=nperseg)
    lf_mask = (f_rr >= 0.04) & (f_rr <= 0.15)
    hf_mask = (f_rr >= 0.15) & (f_rr <= 0.4)
    lf = _trapz(psd_rr[lf_mask], f_rr[lf_mask])
    hf = _trapz(psd_rr[hf_mask], f_rr[hf_mask])
    lf_hf = lf / (hf + 1e-12)

    return {'SDNN': sdnn, 'RMSSD': rmssd, 'pNN50': pnn50, 'LF_HF_Ratio': lf_hf}

# ============================================================
# 模块 4: 信号质量指标 (SQI)
# ============================================================

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2):
    r = r_factor * np.std(x)
    n = len(x)
    if n < m + 2:
        return np.nan

    def _count_matches(template_len):
        templates = np.array([x[i:i + template_len] for i in range(n - template_len)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(dist < r)
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return np.nan
    return -np.log(A / B) if A > 0 else np.nan


def sqi_metrics(raw: np.ndarray, fps: float):
    bp = bandpass_filter(raw, fps)
    ac = remove_dc(raw)

    peaks, props = sig.find_peaks(bp, distance=max(1, int(fps / HR_BAND_HI)), prominence=np.std(bp) * 0.1)
    avg_prominence = np.mean(props['prominences']) if len(peaks) > 0 else 0
    prominence_ratio = avg_prominence / (np.std(bp) + 1e-12)

    ac_corr = np.correlate(bp, bp, mode='full')
    ac_corr = ac_corr[len(bp) - 1:]
    ac_corr /= ac_corr[0] + 1e-12
    min_lag = int(fps / HR_BAND_HI)
    max_lag = int(fps / HR_BAND_LO)
    max_lag = min(max_lag, len(ac_corr) - 1)
    if min_lag < max_lag:
        autocorr_peak = np.max(ac_corr[min_lag:max_lag])
    else:
        autocorr_peak = 0

    zero_crossings = np.sum(np.diff(np.sign(ac)) != 0) / len(ac)
    sampen = sample_entropy(bp[::max(1, len(bp) // 200)])

    return {
        'Peak_Prominence_Ratio': prominence_ratio,
        'Autocorr_Peak': autocorr_peak,
        'Zero_Crossing_Rate': zero_crossings,
        'SampleEntropy': sampen,
    }

# ============================================================
# 模块 5: 两探针对比分析
# ============================================================

def comparison_metrics(raw1: np.ndarray, raw2: np.ndarray, fps: float):
    pcc, _ = pearsonr(raw1, raw2)

    cc = np.correlate(remove_dc(raw1), remove_dc(raw2), mode='full')
    cc /= np.max(np.abs(cc)) + 1e-12
    lags = np.arange(-len(raw1) + 1, len(raw1))
    peak_lag = lags[np.argmax(cc)]
    phase_delay_ms = peak_lag / fps * 1000.0

    n1, n2 = remove_dc(raw1), remove_dc(raw2)
    n1 /= np.std(n1) + 1e-12
    n2 /= np.std(n2) + 1e-12
    # 注意：这是 z-score 标准化后的逐点 RMSE，并非真正的 DTW（动态时间规整）
    rmse_z = np.sqrt(np.mean((n1 - n2) ** 2))

    def _est_bpm(x):
        nperseg = min(256, max(16, len(x) // 4))
        f, p = sig.welch(remove_dc(x), fs=fps, nperseg=nperseg)
        mask = (f >= HR_BAND_LO) & (f <= HR_BAND_HI)
        return f[mask][np.argmax(p[mask])] * 60 if np.any(mask) else 0
    bpm_diff = _est_bpm(raw2) - _est_bpm(raw1)

    delta_snr = peak_snr_db(raw2, fps) - peak_snr_db(raw1, fps)

    nperseg = min(256, max(16, len(raw1) // 4))
    _, psd1 = sig.welch(remove_dc(raw1), fs=fps, nperseg=nperseg)
    _, psd2 = sig.welch(remove_dc(raw2), fs=fps, nperseg=nperseg)
    spectral_cosine = 1.0 - cosine_dist(psd1, psd2)

    return {
        'PCC': pcc,
        'Phase_Delay_ms': phase_delay_ms,
        'Normalized_RMSE_Z': rmse_z,
        'BPM_Diff': bpm_diff,
        'Delta_SNR_dB': delta_snr,
        'Spectral_Cosine_Sim': spectral_cosine,
    }

# ============================================================
# 模块 6: 可视化
# ============================================================

def plot_timeseries(merged: pd.DataFrame, fps: float, name1: str, name2: str, out: str,
                    domain1: str = 'rgb', domain2: str = 'rgb'):
    # Channel labels per domain
    _ch_labels = {
        'rgb': ['C0 (R)', 'C1 (G) ★', 'C2 (B)'],
        'yuv': ['C0 (Y) ★', 'C1 (U)', 'C2 (V)'],
        'raw': ['C0 (RAW) ★', 'C1 (N/A)', 'C2 (N/A)'],
    }
    # 更鲜明的颜色对比：蓝色系 vs 橙红色系
    colors_p1 = ['#1f77b4', '#2ca02c', '#9467bd']  # 蓝、绿、紫
    colors_p2 = ['#ff7f0e', '#d62728', '#e377c2']  # 橙、红、粉

    t = (merged['Frame_ID'].values - merged['Frame_ID'].values[0]) / fps
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i in range(3):
        col = f'ROI_Mean_C{i}'
        v1 = merged[f'{col}_p1'].values.astype(float)
        v2 = merged[f'{col}_p2'].values.astype(float)
        label1 = _ch_labels.get(domain1, _ch_labels['rgb'])[i]
        label2 = _ch_labels.get(domain2, _ch_labels['rgb'])[i]
        ax = axes[i]

        has_v1 = not np.all(np.isnan(v1))
        has_v2 = not np.all(np.isnan(v2))

        if not has_v1 and not has_v2:
            ax.text(0.5, 0.5, f'{label1} / {label2}: N/A', transform=ax.transAxes, ha='center', fontsize=10)
        else:
            if has_v1:
                s1 = bandpass_filter(preprocess(np.where(np.isnan(v1), np.nanmean(v1), v1)), fps)
                ax.plot(t, s1, color=colors_p1[i], alpha=0.9, linewidth=1.2,
                       label=f'{name1} | {domain1.upper()} {label1}')
            else:
                ax.text(0.02, 0.90, f'{name1}: N/A ({domain1.upper()})',
                       transform=ax.transAxes, fontsize=9, color=colors_p1[i])
            if has_v2:
                s2 = bandpass_filter(preprocess(np.where(np.isnan(v2), np.nanmean(v2), v2)), fps)
                ax.plot(t, s2, color=colors_p2[i], alpha=0.9, linewidth=1.2,
                       label=f'{name2} | {domain2.upper()} {label2}')
            else:
                ax.text(0.02, 0.75, f'{name2}: N/A ({domain2.upper()})',
                       transform=ax.transAxes, fontsize=9, color=colors_p2[i])
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        _apply_zh(ax.set_ylabel(f'AC 幅度', fontsize=10))
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    domain_str = f'{domain1.upper()}→{domain2.upper()}' if domain1 != domain2 else domain1.upper()
    _apply_zh(fig.suptitle(f'ROI 通道时域波形对比 (Bandpass {HR_BAND_LO:.2f}-{HR_BAND_HI:.1f} Hz) [{domain_str}]', fontsize=14, weight='bold'))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'timeseries_comparison.png'), dpi=150)
    plt.close(fig)


def _get_signal_for_plot(merged: pd.DataFrame, suffix: str, domain: str) -> np.ndarray:
    raw, _, _ = _pick_channel_from(merged, suffix, domain)
    raw = np.where(np.isnan(raw), np.nanmean(raw) if not np.all(np.isnan(raw)) else 0, raw)
    return preprocess(raw)


def _sanitize_name(name: str) -> str:
    return _display_probe_name(name).replace(os.sep, '_').replace(' ', '_')


def _raw_channel_output_dir(out: str) -> str:
    raw_dir = os.path.join(out, 'raw_channels')
    os.makedirs(raw_dir, exist_ok=True)
    return raw_dir


def _get_raw_multichannel_signals(merged: pd.DataFrame, suffix: str):
    """返回 RAW 多通道时序与对应 AC Delta，仅在列存在且有效时输出。"""
    signals = {}
    for key, mean_col, delta_col, color in RAW_MULTI_CHANNELS:
        full_mean = f'{mean_col}{suffix}'
        if full_mean not in merged.columns:
            continue
        series = merged[full_mean].values.astype(float)
        if np.isfinite(series).sum() <= 10:
            continue
        raw = np.where(np.isnan(series), np.nanmean(series), series)
        full_delta = f'{delta_col}{suffix}'
        if full_delta in merged.columns:
            ac_delta = merged[full_delta].values.astype(float)
            ac_delta = np.nan_to_num(ac_delta, nan=0.0)
        else:
            ac_delta = np.zeros(len(raw), dtype=float)
        signals[key] = {
            'raw': raw,
            'processed': preprocess(raw),
            'ac_delta': ac_delta,
            'color': color,
            'mean_col': mean_col,
            'delta_col': delta_col,
        }
    return signals


def _collect_single_signal_metrics(raw_signal: np.ndarray, ac_delta: np.ndarray, fps: float):
    processed = preprocess(raw_signal)
    ac = remove_dc(processed)
    metrics = {}
    metrics.update(time_domain_metrics(processed, ac, ac_delta, fps))
    fd, _, _ = freq_domain_metrics(processed, fps)
    metrics.update(fd)
    rr, _ = extract_rr_intervals(processed, fps)
    metrics.update(hrv_metrics(rr, fps))
    metrics.update(sqi_metrics(processed, fps))
    dc = np.mean(raw_signal)
    ac_amplitude = np.std(raw_signal)
    metrics['AC_DC_Ratio'] = ac_amplitude / dc if dc > 1e-12 else 0
    return metrics


def _store_raw_multichannel_metrics(all_metrics: dict, probe_name: str, raw_signals: dict, fps: float):
    for key, payload in raw_signals.items():
        metrics = _collect_single_signal_metrics(payload['raw'], payload['ac_delta'], fps)
        for metric_name, value in metrics.items():
            all_metrics[f'{probe_name}_{key}_{metric_name}'] = value


def _safe_pearson(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or len(y) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(pearsonr(x, y)[0])


def _estimate_peak_bpm(x: np.ndarray, fps: float):
    nperseg = min(256, max(16, len(x) // 4))
    f, p = sig.welch(remove_dc(x), fs=fps, nperseg=nperseg)
    mask = (f >= HR_BAND_LO) & (f <= HR_BAND_HI)
    if not np.any(mask) or np.max(p[mask]) <= 0:
        return np.nan
    return float(f[mask][np.argmax(p[mask])] * 60.0)


def _pair_metric_name(a: str, b: str, suffix: str) -> str:
    return f'{a}_{b}_{suffix}'


def _store_raw_consistency_metrics(all_metrics: dict, probe_name: str, raw_signals: dict, fps: float):
    g1 = raw_signals.get('RAW_G1')
    g2 = raw_signals.get('RAW_G2')
    if g1 and g2:
        g1_proc = g1['processed']
        g2_proc = g2['processed']
        cmp = comparison_metrics(g1_proc, g2_proc, fps)
        all_metrics[f'{probe_name}_RAW_G1_G2_PCC'] = cmp['PCC']
        all_metrics[f'{probe_name}_RAW_G1_G2_Phase_Delay_ms'] = cmp['Phase_Delay_ms']
        all_metrics[f'{probe_name}_RAW_G1_G2_Normalized_RMSE_Z'] = cmp['Normalized_RMSE_Z']
        all_metrics[f'{probe_name}_RAW_G1_G2_BPM_Diff'] = cmp['BPM_Diff']
        all_metrics[f'{probe_name}_RAW_G1_G2_Delta_SNR_dB'] = cmp['Delta_SNR_dB']
        all_metrics[f'{probe_name}_RAW_G1_G2_Spectral_Cosine_Sim'] = cmp['Spectral_Cosine_Sim']
        all_metrics[f'{probe_name}_RAW_G1_G2_AC_Delta_Mean_Diff'] = float(
            np.mean(g2['ac_delta']) - np.mean(g1['ac_delta'])
        )

    for key_a, key_b in combinations(raw_signals.keys(), 2):
        proc_a = raw_signals[key_a]['processed']
        proc_b = raw_signals[key_b]['processed']
        raw_a = raw_signals[key_a]['raw']
        raw_b = raw_signals[key_b]['raw']
        all_metrics[f'{probe_name}_{_pair_metric_name(key_a, key_b, "Corr")}'] = _safe_pearson(proc_a, proc_b)
        bpm_a = _estimate_peak_bpm(proc_a, fps)
        bpm_b = _estimate_peak_bpm(proc_b, fps)
        if np.isfinite(bpm_a) and np.isfinite(bpm_b):
            all_metrics[f'{probe_name}_{_pair_metric_name(key_a, key_b, "BPM_Diff")}'] = float(bpm_b - bpm_a)
        else:
            all_metrics[f'{probe_name}_{_pair_metric_name(key_a, key_b, "BPM_Diff")}'] = np.nan
        mean_a = np.mean(raw_a)
        mean_b = np.mean(raw_b)
        if abs(mean_a) > 1e-12:
            all_metrics[f'{probe_name}_{_pair_metric_name(key_a, key_b, "Mean_Ratio")}'] = float(mean_b / mean_a)
        else:
            all_metrics[f'{probe_name}_{_pair_metric_name(key_a, key_b, "Mean_Ratio")}'] = np.nan


def plot_raw_channel_timeseries(raw_signals: dict, fps: float, name: str, out: str):
    if not raw_signals:
        return
    raw_dir = _raw_channel_output_dir(out)
    t = np.arange(len(next(iter(raw_signals.values()))['processed'])) / fps
    fig, ax = plt.subplots(figsize=(14, 5))
    for key, payload in raw_signals.items():
        ax.plot(t, bandpass_filter(payload['processed'], fps), label=key, color=payload['color'], linewidth=1.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AC amplitude')
    ax.set_title(f'{_display_probe_name(name)} RAW channel timeseries')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(ncol=5, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(raw_dir, f'timeseries_{_sanitize_name(name)}.png'), dpi=150)
    plt.close(fig)


def plot_raw_channel_psd(raw_signals: dict, fps: float, name: str, out: str):
    if not raw_signals:
        return
    raw_dir = _raw_channel_output_dir(out)
    fig, ax = plt.subplots(figsize=(14, 5))
    for key, payload in raw_signals.items():
        raw = payload['processed']
        nperseg = min(256, max(16, len(raw) // 4))
        f, psd = sig.welch(remove_dc(raw), fs=fps, nperseg=nperseg)
        mask = f <= HR_BAND_HI
        ax.semilogy(f[mask], psd[mask], label=key, color=payload['color'], linewidth=1.6)
    ax.axvspan(HR_BAND_LO, HR_BAND_HI, alpha=0.12, color='red')
    ax.set_xlim(0, HR_BAND_HI)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'{_display_probe_name(name)} RAW channel PSD')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(ncol=5, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(raw_dir, f'psd_{_sanitize_name(name)}.png'), dpi=150)
    plt.close(fig)


def plot_raw_g1g2_overlay(raw_signals: dict, fps: float, name: str, out: str):
    g1 = raw_signals.get('RAW_G1')
    g2 = raw_signals.get('RAW_G2')
    if not g1 or not g2:
        return
    raw_dir = _raw_channel_output_dir(out)
    t = np.arange(len(g1['processed'])) / fps
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t, bandpass_filter(g1['processed'], fps), color=g1['color'], linewidth=1.4, label='RAW_G1')
    axes[0].plot(t, bandpass_filter(g2['processed'], fps), color=g2['color'], linewidth=1.2, alpha=0.9, label='RAW_G2')
    axes[0].set_ylabel('AC amplitude')
    axes[0].set_title(f'{_display_probe_name(name)} G1/G2 consistency')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(framealpha=0.9)

    diff = bandpass_filter(g2['processed'], fps) - bandpass_filter(g1['processed'], fps)
    axes[1].plot(t, diff, color='#444444', linewidth=1.1, label='G2 - G1')
    axes[1].axhline(0, color='k', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Delta')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(raw_dir, f'g1g2_consistency_{_sanitize_name(name)}.png'), dpi=150)
    plt.close(fig)


def plot_raw_channel_correlation_heatmap(raw_signals: dict, name: str, out: str):
    if len(raw_signals) < 2:
        return
    raw_dir = _raw_channel_output_dir(out)
    keys = [key for key in RAW_MULTI_KEYS if key in raw_signals]
    matrix = np.full((len(keys), len(keys)), np.nan, dtype=float)
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            if i == j:
                matrix[i, j] = 1.0
            elif j > i:
                matrix[i, j] = _safe_pearson(raw_signals[key_i]['processed'], raw_signals[key_j]['processed'])
                matrix[j, i] = matrix[i, j]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap='coolwarm')
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha='right')
    ax.set_yticklabels(keys)
    ax.set_title(f'{_display_probe_name(name)} RAW channel correlation')
    for i in range(len(keys)):
        for j in range(len(keys)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9, color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    fig.tight_layout()
    fig.savefig(os.path.join(raw_dir, f'corr_heatmap_{_sanitize_name(name)}.png'), dpi=150)
    plt.close(fig)


def plot_psd_cross(merged: pd.DataFrame, fps: float, name1: str, name2: str, out: str,
                   domain1: str, domain2: str, ch_label: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    # 计算 PSD 和 SNR
    snr_data = {}
    peak_data = {}
    for suffix, name, color, domain in [
        ('_p1', name1, '#1f77b4', domain1),
        ('_p2', name2, '#ff7f0e', domain2)
    ]:
        raw = _get_signal_for_plot(merged, suffix, domain)
        nperseg = min(256, max(16, len(raw) // 4))
        f, psd = sig.welch(remove_dc(raw), fs=fps, nperseg=nperseg)

        mask_4hz = f <= HR_BAND_HI
        f_plot = f[mask_4hz]
        psd_plot = psd[mask_4hz]

        hr_mask = (f >= HR_BAND_LO) & (f <= HR_BAND_HI)
        if np.any(hr_mask) and np.max(psd[hr_mask]) > 0:
            peak_idx = np.argmax(psd[hr_mask])
            peak_freq = f[hr_mask][peak_idx]
            peak_bpm = peak_freq * 60
            peak_psd = psd[hr_mask][peak_idx]
            peak_data[suffix] = (peak_freq, peak_bpm, peak_psd)
        else:
            peak_data[suffix] = (0, 0, 0)

        # SNR 使用统一公式
        snr_data[suffix] = peak_snr_db(raw, fps)

        ax.semilogy(f_plot, psd_plot, color=color, linewidth=2, label=f'{name} [{domain.upper()}]')

    # 标注峰值（错开位置避免重叠）
    annotate_offsets = [(10, 20), (10, -30)]
    for idx, suffix in enumerate(['_p1', '_p2']):
        if peak_data[suffix][0] > 0:
            pf, pb, pp = peak_data[suffix]
            color = '#1f77b4' if suffix == '_p1' else '#ff7f0e'
            ax.plot(pf, pp, 'o', color=color, markersize=8)
            ax.annotate(f'{pf:.2f}Hz\n{pb:.1f}BPM',
                       xy=(pf, pp), xytext=annotate_offsets[idx], textcoords='offset points',
                       fontsize=9, color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1))

    ax.axvspan(HR_BAND_LO, HR_BAND_HI, alpha=0.15, color='red', label='HR Band')

    # 右上角显示 SNR（处理无效值）
    snr1 = snr_data.get('_p1', 0)
    snr2 = snr_data.get('_p2', 0)
    delta_snr = snr2 - snr1
    snr_text = f"SNR_p1: {snr1:.2f} dB\nSNR_p2: {snr2:.2f} dB\nΔSNR: {delta_snr:.2f} dB"
    ax.text(0.98, 0.97, snr_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('PSD', fontsize=11)
    _apply_zh(ax.set_title(f'PSD 对比 [{ch_label}]', fontsize=13, weight='bold'))
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, HR_BAND_HI)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'psd_comparison.png'), dpi=150)
    plt.close(fig)


def plot_spectrogram_cross(merged: pd.DataFrame, fps: float, name1: str, name2: str, out: str,
                           domain1: str, domain2: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    nperseg = min(128, max(16, len(merged) // 4))

    for ax, suffix, name, domain in zip(axes, ['_p1', '_p2'], [name1, name2], [domain1, domain2]):
        raw = _get_signal_for_plot(merged, suffix, domain)
        f, t_spec, Sxx = sig.spectrogram(remove_dc(raw), fs=fps, nperseg=nperseg, noverlap=nperseg // 2)
        mask = f <= 5.0
        ax.pcolormesh(t_spec, f[mask], 10 * np.log10(Sxx[mask] + 1e-12), shading='gouraud', cmap='inferno')
        ax.axhline(HR_BAND_LO, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(HR_BAND_HI, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(f'{name} [{domain}] Spectrogram')

    axes[-1].set_xlabel('Time (frames / fps)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'spectrogram.png'), dpi=150)
    plt.close(fig)


def plot_poincare(rr1: np.ndarray, rr2: np.ndarray, name1: str, name2: str, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rr, name in zip(axes, [rr1, rr2], [name1, name2]):
        if len(rr) > 1:
            ax.scatter(rr[:-1], rr[1:], s=10, alpha=0.6)
            lim = [min(rr.min(), rr[1:].min()) * 0.9, max(rr.max(), rr[:-1].max()) * 1.1]
            ax.plot(lim, lim, 'k--', alpha=0.3)
        ax.set_xlabel('RR_n (ms)')
        ax.set_ylabel('RR_n+1 (ms)')
        ax.set_title(f'{name} Poincaré')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out, 'hrv_poincare.png'), dpi=150)
    plt.close(fig)

# ============================================================
# 主流程
# ============================================================

def _domain_channel(domain: str) -> tuple:
    """返回 (col_idx, label)"""
    if domain == 'raw':
        return None, 'RAW_G(Main)'
    elif domain == 'yuv':
        return 0, 'C0(Y/Luma)'
    else:
        return 1, 'C1(G)'


def _pick_series_by_candidates(merged: pd.DataFrame, suffix: str, candidates):
    """按候选列顺序选择有效时序，返回 (values, label, column_name)。"""
    for col_name, label in candidates:
        full_col = f'{col_name}{suffix}'
        if full_col not in merged.columns:
            continue
        vals = merged[full_col]
        if vals.notna().sum() > 10:
            return vals.values.astype(float), label, col_name
    fallback_col = f'ROI_Mean_C0{suffix}'
    if fallback_col in merged.columns:
        return merged[fallback_col].values.astype(float), 'C0(fallback)', 'ROI_Mean_C0'
    raise KeyError(f'No usable signal columns found for suffix={suffix}.')


def _pick_channel_from(merged: pd.DataFrame, suffix: str, domain: str):
    if domain == 'raw':
        candidates = [
            ('RAW_G_Mean', 'RAW_G(Bayer-aware)'),
            ('ROI_Mean_C0', 'C0(RAW fallback)'),
        ]
    elif domain == 'yuv':
        candidates = [
            ('ROI_Mean_C0', 'C0(Y/Luma)'),
            ('ROI_Mean_C1', 'C1(U fallback)'),
            ('ROI_Mean_C2', 'C2(V fallback)'),
        ]
    else:
        candidates = [
            ('ROI_Mean_C1', 'C1(G)'),
            ('ROI_Mean_C0', 'C0(R fallback)'),
            ('ROI_Mean_C2', 'C2(B fallback)'),
        ]
    return _pick_series_by_candidates(merged, suffix, candidates)


def _pick_ac_delta_from(merged: pd.DataFrame, suffix: str, signal_col: str):
    """为主分析通道选择最匹配的 AC Delta 列。"""
    if signal_col.startswith('RAW_') and signal_col.endswith('_Mean'):
        raw_delta_col = f"{signal_col[:-5]}_AC_Delta{suffix}"
        if raw_delta_col in merged.columns:
            vals = merged[raw_delta_col].values.astype(float)
            if np.isfinite(vals).sum() > 10:
                return np.nan_to_num(vals, nan=0.0), raw_delta_col[:-len(suffix)]
    roi_delta_col = f'ROI_AC_Delta{suffix}'
    if roi_delta_col in merged.columns:
        vals = merged[roi_delta_col].values.astype(float)
        return np.nan_to_num(vals, nan=0.0), 'ROI_AC_Delta'
    return np.zeros(len(merged), dtype=float), 'zero_fallback'


def _display_probe_name(name: str) -> str:
    return name[:-11] if name.endswith('_timeseries') else name


def _metric_value(all_metrics: dict, owner: str, metric: str):
    return all_metrics.get(f'{owner}_{metric}')


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, bool)) and not isinstance(value, bool)


def _format_value(metric: str, value, signed: bool = False) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return 'True' if value else 'False'
    if not np.isfinite(value):
        return ''

    precision = 4
    if metric.endswith('_Count') or metric in {'Valid_GT_Window_Count', 'Valid_GT_SNR_Window_Count'}:
        precision = 0
    elif metric in {
        'BPM_Estimate',
        'BPM_Diff',
        'BPM_MAE',
        'BPM_RMSE',
        'BPM_MAXError',
        'BPM_Bias',
        'GT_BPM_Mean',
        'GT_BPM_Min',
        'GT_BPM_Max',
    }:
        precision = 2
    elif metric in {
        'GT_SNR_dB',
        'GT_SNR_Median_dB',
        'GT_SNR_Std_dB',
        'GT_Window_Size_s',
        'GT_Stride_s',
        'GT_Probe_Time_Offset_s',
        'GT_First_Window_Center_s',
        'GT_Last_Window_Center_s',
    }:
        precision = 3
    elif metric in {'Phase_Delay_ms'}:
        precision = 3

    fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
    return fmt.format(float(value))


def _format_delta(metric: str, value1, value2) -> str:
    if value1 in (None, '') or value2 in (None, ''):
        if value1 in (None, '') and value2 in (None, ''):
            return ''
        return 'N/A'
    if _is_number(value1) and _is_number(value2):
        return _format_value(metric, float(value2) - float(value1), signed=True)
    if value1 == value2:
        return '-'
    return f'{value1}->{value2}'


def _delta_label(metric: str) -> str:
    labels = {
        'SNR_dB': 'Delta_SNR',
        'BPM_Estimate': 'BPM_Diff',
        'GT_SNR_dB': 'Delta_GT_SNR',
        'GT_SNR_Median_dB': 'Delta_GT_SNR_Median',
        'BPM_MAE': 'Delta_MAE',
        'BPM_RMSE': 'Delta_RMSE',
        'BPM_MAXError': 'Delta_MAX',
        'BPM_Bias': 'Delta_Bias',
        'BPM_PCC': 'Delta_PCC',
        'BPM_CCC': 'Delta_CCC',
    }
    return labels.get(metric, '')


def _display_metric(metric: str) -> str:
    labels = {
        'PCC': 'PCC (波形相似度)',
        'RAW_G1_G2_PCC': 'RAW_G1_G2_PCC',
        'RAW_G1_G2_Phase_Delay_ms': 'RAW_G1_G2_Phase_Delay_ms',
        'RAW_G1_G2_Normalized_RMSE_Z': 'RAW_G1_G2_Normalized_RMSE_Z',
        'RAW_G1_G2_BPM_Diff': 'RAW_G1_G2_BPM_Diff',
        'RAW_G1_G2_Delta_SNR_dB': 'RAW_G1_G2_Delta_SNR_dB',
        'RAW_G1_G2_Spectral_Cosine_Sim': 'RAW_G1_G2_Spectral_Cosine_Sim',
        'RAW_G1_G2_AC_Delta_Mean_Diff': 'RAW_G1_G2_AC_Delta_Mean_Diff',
        'GT_SNR_dB': 'GT_SNR_dB (pyVHR)',
        'BPM_MAE': 'BPM_MAE_vs_GT',
        'BPM_RMSE': 'BPM_RMSE_vs_GT',
        'BPM_MAXError': 'BPM_MAXError_vs_GT',
        'BPM_Bias': 'BPM_Bias_EstMinusGT',
        'BPM_PCC': 'BPM_PCC_vs_GT',
        'BPM_CCC': 'BPM_CCC_vs_GT',
    }
    return labels.get(metric, metric)


def _section_title(category_key: str) -> str:
    titles = {
        'basic': '--- 基础信息区 ---',
        'core': '--- 核心判决区 ---',
        'time_quality': '--- 时域波形质量区 ---',
        'freq_quality': '--- 频域纯净度区 ---',
        'hrv': '--- HRV 生理区 ---',
        'color': '--- 通道与色彩区 ---',
        'raw_multi': '--- RAW 多通道分析 ---',
        'raw_consistency': '--- RAW 一致性与通道间比较 ---',
        'gt': '--- GT 参考对比区 ---',
        'compare': '--- 跨探针对比 ---',
    }
    return titles[category_key]


def _append_section_rows(rows, category_key: str, metrics, probe1: str, probe2: str):
    valid_metrics = [item for item in metrics if not (item[1] is None and item[2] is None and item[3] is None)]
    if not valid_metrics:
        return
    rows.append({
        'Metric': _section_title(category_key),
        f'P1: {probe1}': '-',
        f'P2: {probe2}': '-',
        'Compare/Delta': '-',
    })
    for metric, v1, v2, delta_override in valid_metrics:
        delta_value = delta_override if delta_override is not None else _format_delta(metric, v1, v2)
        label = _delta_label(metric)
        if label and delta_value not in ('', '-'):
            delta_value = f'{delta_value} ({label})'
        rows.append({
            'Metric': _display_metric(metric),
            f'P1: {probe1}': _format_value(metric, v1),
            f'P2: {probe2}': _format_value(metric, v2),
            'Compare/Delta': delta_value if delta_value not in ('', None) else 'N/A',
        })


def _build_raw_multichannel_rows(all_metrics: dict, name1: str, name2: str):
    probe1 = _display_probe_name(name1)
    probe2 = _display_probe_name(name2)
    metrics = []
    focus_metrics = ['SNR_dB', 'BPM_Estimate', 'AC_DC_Ratio', 'Autocorr_Peak']
    for raw_key in ['RAW_R', 'RAW_G1', 'RAW_G2', 'RAW_G', 'RAW_B']:
        for metric in focus_metrics:
            v1 = all_metrics.get(f'{name1}_{raw_key}_{metric}')
            v2 = all_metrics.get(f'{name2}_{raw_key}_{metric}')
            if v1 is None and v2 is None:
                continue
            metrics.append((f'{raw_key}_{metric}', v1, v2, None))
    if not metrics:
        return []
    rows = []
    _append_section_rows(rows, 'raw_multi', metrics, probe1, probe2)
    return rows


def _build_raw_consistency_rows(all_metrics: dict, name1: str, name2: str):
    probe1 = _display_probe_name(name1)
    probe2 = _display_probe_name(name2)
    metrics = []
    focus_metrics = [
        'RAW_G1_G2_PCC',
        'RAW_G1_G2_Phase_Delay_ms',
        'RAW_G1_G2_Normalized_RMSE_Z',
        'RAW_G1_G2_BPM_Diff',
        'RAW_G1_G2_Delta_SNR_dB',
        'RAW_G1_G2_Spectral_Cosine_Sim',
        'RAW_G1_G2_AC_Delta_Mean_Diff',
        _pair_metric_name('RAW_R', 'RAW_G', 'Corr'),
        _pair_metric_name('RAW_B', 'RAW_G', 'Corr'),
        _pair_metric_name('RAW_R', 'RAW_B', 'Corr'),
    ]
    for metric in focus_metrics:
        v1 = all_metrics.get(f'{name1}_{metric}')
        v2 = all_metrics.get(f'{name2}_{metric}')
        if v1 is None and v2 is None:
            continue
        metrics.append((metric, v1, v2, None))
    if not metrics:
        return []
    rows = []
    _append_section_rows(rows, 'raw_consistency', metrics, probe1, probe2)
    return rows


def _build_gt_rows(all_metrics: dict, name1: str, name2: str):
    if not all_metrics.get('gt_enabled'):
        return []
    probe1 = _display_probe_name(name1)
    probe2 = _display_probe_name(name2)
    metrics = [
        ('GT_Source', all_metrics.get('gt_source'), all_metrics.get('gt_source'), '-'),
        ('GT_Subject', all_metrics.get('gt_subject'), all_metrics.get('gt_subject'), '-'),
        ('GT_Probe_Time_Mode', _metric_value(all_metrics, name1, 'GT_Probe_Time_Mode'),
         _metric_value(all_metrics, name2, 'GT_Probe_Time_Mode'), '-'),
        ('GT_Probe_Time_Offset_s', _metric_value(all_metrics, name1, 'GT_Probe_Time_Offset_s'),
         _metric_value(all_metrics, name2, 'GT_Probe_Time_Offset_s'), '-'),
        ('GT_Window_Size_s', _metric_value(all_metrics, name1, 'GT_Window_Size_s'),
         _metric_value(all_metrics, name2, 'GT_Window_Size_s'), '-'),
        ('GT_Stride_s', _metric_value(all_metrics, name1, 'GT_Stride_s'),
         _metric_value(all_metrics, name2, 'GT_Stride_s'), '-'),
        ('Valid_GT_Window_Count', _metric_value(all_metrics, name1, 'Valid_GT_Window_Count'),
         _metric_value(all_metrics, name2, 'Valid_GT_Window_Count'), None),
        ('Valid_GT_SNR_Window_Count', _metric_value(all_metrics, name1, 'Valid_GT_SNR_Window_Count'),
         _metric_value(all_metrics, name2, 'Valid_GT_SNR_Window_Count'), None),
        ('GT_SNR_dB', _metric_value(all_metrics, name1, 'GT_SNR_dB'),
         _metric_value(all_metrics, name2, 'GT_SNR_dB'),
         _format_value('GT_SNR_dB', all_metrics.get('Compare_Delta_GT_SNR_dB'), signed=True)),
        ('GT_SNR_Median_dB', _metric_value(all_metrics, name1, 'GT_SNR_Median_dB'),
         _metric_value(all_metrics, name2, 'GT_SNR_Median_dB'),
         _format_value('GT_SNR_Median_dB', all_metrics.get('Compare_Delta_GT_SNR_Median_dB'), signed=True)),
        ('GT_SNR_Std_dB', _metric_value(all_metrics, name1, 'GT_SNR_Std_dB'),
         _metric_value(all_metrics, name2, 'GT_SNR_Std_dB'), None),
        ('BPM_MAE', _metric_value(all_metrics, name1, 'BPM_MAE'),
         _metric_value(all_metrics, name2, 'BPM_MAE'),
         _format_value('BPM_MAE', all_metrics.get('Compare_Delta_BPM_MAE'), signed=True)),
        ('BPM_RMSE', _metric_value(all_metrics, name1, 'BPM_RMSE'),
         _metric_value(all_metrics, name2, 'BPM_RMSE'),
         _format_value('BPM_RMSE', all_metrics.get('Compare_Delta_BPM_RMSE'), signed=True)),
        ('BPM_MAXError', _metric_value(all_metrics, name1, 'BPM_MAXError'),
         _metric_value(all_metrics, name2, 'BPM_MAXError'),
         _format_value('BPM_MAXError', all_metrics.get('Compare_Delta_BPM_MAXError'), signed=True)),
        ('BPM_Bias', _metric_value(all_metrics, name1, 'BPM_Bias'),
         _metric_value(all_metrics, name2, 'BPM_Bias'),
         _format_value('BPM_Bias', all_metrics.get('Compare_Delta_BPM_Bias'), signed=True)),
        ('BPM_PCC', _metric_value(all_metrics, name1, 'BPM_PCC'),
         _metric_value(all_metrics, name2, 'BPM_PCC'),
         _format_value('BPM_PCC', all_metrics.get('Compare_Delta_BPM_PCC'), signed=True)),
        ('BPM_CCC', _metric_value(all_metrics, name1, 'BPM_CCC'),
         _metric_value(all_metrics, name2, 'BPM_CCC'),
         _format_value('BPM_CCC', all_metrics.get('Compare_Delta_BPM_CCC'), signed=True)),
        ('GT_BPM_Mean', _metric_value(all_metrics, name1, 'GT_BPM_Mean'),
         _metric_value(all_metrics, name2, 'GT_BPM_Mean'), None),
        ('GT_BPM_Min', _metric_value(all_metrics, name1, 'GT_BPM_Min'),
         _metric_value(all_metrics, name2, 'GT_BPM_Min'), None),
        ('GT_BPM_Max', _metric_value(all_metrics, name1, 'GT_BPM_Max'),
         _metric_value(all_metrics, name2, 'GT_BPM_Max'), None),
        ('GT_First_Window_Center_s', _metric_value(all_metrics, name1, 'GT_First_Window_Center_s'),
         _metric_value(all_metrics, name2, 'GT_First_Window_Center_s'), None),
        ('GT_Last_Window_Center_s', _metric_value(all_metrics, name1, 'GT_Last_Window_Center_s'),
         _metric_value(all_metrics, name2, 'GT_Last_Window_Center_s'), None),
    ]
    rows = []
    _append_section_rows(rows, 'gt', metrics, probe1, probe2)
    return rows


def _build_summary_table(all_metrics: dict, name1: str, name2: str) -> pd.DataFrame:
    probe1 = _display_probe_name(name1)
    probe2 = _display_probe_name(name2)

    rows = []

    basic_metrics = [
        ('Domain', all_metrics.get('domain_p1'), all_metrics.get('domain_p2'), None),
        ('Primary_Channel', _metric_value(all_metrics, name1, 'Primary_Channel'), _metric_value(all_metrics, name2, 'Primary_Channel'), None),
    ]
    raw_mode_1 = all_metrics.get('raw_mode_p1')
    raw_mode_2 = all_metrics.get('raw_mode_p2')
    bayer_1 = all_metrics.get('raw_bayer_pattern_p1')
    bayer_2 = all_metrics.get('raw_bayer_pattern_p2')
    if raw_mode_1 == 'bayer_aware' or raw_mode_2 == 'bayer_aware':
        basic_metrics.append(('Raw_Mode', raw_mode_1, raw_mode_2, None))
    if bayer_1 or bayer_2:
        basic_metrics.append(('Bayer_Pattern', bayer_1 or '-', bayer_2 or '-', None))

    core_metrics = [
        ('SNR_dB', _metric_value(all_metrics, name1, 'SNR_dB'), _metric_value(all_metrics, name2, 'SNR_dB'), None),
        ('BPM_Estimate', _metric_value(all_metrics, name1, 'BPM_Estimate'), _metric_value(all_metrics, name2, 'BPM_Estimate'), None),
    ]
    time_quality_metrics = [
        ('AC_DC_Ratio', _metric_value(all_metrics, name1, 'AC_DC_Ratio'), _metric_value(all_metrics, name2, 'AC_DC_Ratio'), None),
        ('AC_PeakToPeak', _metric_value(all_metrics, name1, 'AC_PeakToPeak'), _metric_value(all_metrics, name2, 'AC_PeakToPeak'), None),
        ('Autocorr_Peak', _metric_value(all_metrics, name1, 'Autocorr_Peak'), _metric_value(all_metrics, name2, 'Autocorr_Peak'), None),
    ]
    freq_quality_metrics = [
        ('HR_Band_Ratio', _metric_value(all_metrics, name1, 'HR_Band_Ratio'), _metric_value(all_metrics, name2, 'HR_Band_Ratio'), None),
        ('Spectral_Entropy', _metric_value(all_metrics, name1, 'Spectral_Entropy'), _metric_value(all_metrics, name2, 'Spectral_Entropy'), None),
        ('Harmonic_Ratio', _metric_value(all_metrics, name1, 'Harmonic_Ratio'), _metric_value(all_metrics, name2, 'Harmonic_Ratio'), None),
        ('Peak_Freq_Hz', _metric_value(all_metrics, name1, 'Peak_Freq_Hz'), _metric_value(all_metrics, name2, 'Peak_Freq_Hz'), None),
        ('Peak_Prominence_Ratio', _metric_value(all_metrics, name1, 'Peak_Prominence_Ratio'), _metric_value(all_metrics, name2, 'Peak_Prominence_Ratio'), None),
        ('SampleEntropy', _metric_value(all_metrics, name1, 'SampleEntropy'), _metric_value(all_metrics, name2, 'SampleEntropy'), None),
    ]
    hrv_metrics = [
        ('SDNN', _metric_value(all_metrics, name1, 'SDNN'), _metric_value(all_metrics, name2, 'SDNN'), None),
        ('RMSSD', _metric_value(all_metrics, name1, 'RMSSD'), _metric_value(all_metrics, name2, 'RMSSD'), None),
        ('pNN50', _metric_value(all_metrics, name1, 'pNN50'), _metric_value(all_metrics, name2, 'pNN50'), None),
        ('LF_HF_Ratio', _metric_value(all_metrics, name1, 'LF_HF_Ratio'), _metric_value(all_metrics, name2, 'LF_HF_Ratio'), None),
    ]
    color_metrics = [
        ('G_R_Ratio', _metric_value(all_metrics, name1, 'G_R_Ratio'), _metric_value(all_metrics, name2, 'G_R_Ratio'), None),
        ('G_B_Ratio', _metric_value(all_metrics, name1, 'G_B_Ratio'), _metric_value(all_metrics, name2, 'G_B_Ratio'), None),
        ('Channel_Corr_GR', _metric_value(all_metrics, name1, 'Channel_Corr_GR'), _metric_value(all_metrics, name2, 'Channel_Corr_GR'), None),
        ('Channel_Corr_GB', _metric_value(all_metrics, name1, 'Channel_Corr_GB'), _metric_value(all_metrics, name2, 'Channel_Corr_GB'), None),
    ]
    compare_metrics = [
        ('PCC', '-', '-', _format_value('PCC', all_metrics.get('Compare_PCC'))),
        ('Phase_Delay_ms', '-', '-', _format_value('Phase_Delay_ms', all_metrics.get('Compare_Phase_Delay_ms'))),
        ('Normalized_RMSE_Z', '-', '-', _format_value('Normalized_RMSE_Z', all_metrics.get('Compare_Normalized_RMSE_Z'))),
        ('BPM_Diff', '-', '-', _format_value('BPM_Diff', all_metrics.get('Compare_BPM_Diff'))),
        ('Delta_SNR_dB', '-', '-', _format_value('Delta_SNR_dB', all_metrics.get('Compare_Delta_SNR_dB'))),
        ('Spectral_Cosine_Sim', '-', '-', _format_value('Spectral_Cosine_Sim', all_metrics.get('Compare_Spectral_Cosine_Sim'))),
    ]

    _append_section_rows(rows, 'basic', basic_metrics, probe1, probe2)
    _append_section_rows(rows, 'core', core_metrics, probe1, probe2)
    _append_section_rows(rows, 'time_quality', time_quality_metrics, probe1, probe2)
    _append_section_rows(rows, 'freq_quality', freq_quality_metrics, probe1, probe2)
    if any(v1 is not None or v2 is not None for _, v1, v2, _ in hrv_metrics):
        _append_section_rows(rows, 'hrv', hrv_metrics, probe1, probe2)
    if any(v1 is not None or v2 is not None for _, v1, v2, _ in color_metrics):
        _append_section_rows(rows, 'color', color_metrics, probe1, probe2)
    rows.extend(_build_raw_multichannel_rows(all_metrics, name1, name2))
    rows.extend(_build_raw_consistency_rows(all_metrics, name1, name2))
    rows.extend(_build_gt_rows(all_metrics, name1, name2))
    _append_section_rows(rows, 'compare', compare_metrics, probe1, probe2)

    return pd.DataFrame(rows)


def run_analysis(csv1: str, csv2: str, fps: float, output_dir: str, domain_arg: str = 'auto',
                 gt_ref=None, gt_window_size: float = GT_DEFAULT_WINDOW_SIZE,
                 gt_stride: float = GT_DEFAULT_STRIDE,
                 probe_time_mode: str = 'absolute_frame_id',
                 probe_time_offset_sec: float = 0.0):
    name1 = os.path.splitext(os.path.basename(csv1))[0]
    name2 = os.path.splitext(os.path.basename(csv2))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载数据: {csv1} vs {csv2}")
    merged = load_and_align(csv1, csv2)
    print(f"对齐后帧数: {len(merged)}")
    if len(merged) == 0:
        print("错误：对齐后无有效帧，请检查两个CSV的Frame_ID是否有交集。")
        return

    # 每个探针独立检测域（仅读 ROI_Mean_C1 列，避免重复全表读取）
    meta1 = load_probe_meta(csv1)
    meta2 = load_probe_meta(csv2)
    if domain_arg == 'auto':
        def _read_c1(csv_path: str) -> pd.DataFrame:
            try:
                return pd.read_csv(csv_path, usecols=['ROI_Mean_C1'])
            except (ValueError, KeyError):
                return pd.DataFrame()
        domain1 = detect_domain(csv1, _read_c1(csv1), meta1)
        domain2 = detect_domain(csv2, _read_c1(csv2), meta2)
    else:
        domain1 = domain2 = domain_arg

    is_cross_domain = (domain1 != domain2)
    domain_display = f"{domain1.upper()}→{domain2.upper()}" if is_cross_domain else domain1.upper()
    print(f"  探针1域: {domain1}  探针2域: {domain2}" + (" [跨域对比]" if is_cross_domain else ""))

    all_metrics = {}
    all_metrics['domain_p1'] = domain1
    all_metrics['domain_p2'] = domain2
    all_metrics['raw_mode_p1'] = meta1.get('raw_mode', 'legacy')
    all_metrics['raw_mode_p2'] = meta2.get('raw_mode', 'legacy')
    all_metrics['raw_bayer_pattern_p1'] = meta1.get('raw_bayer_pattern', '')
    all_metrics['raw_bayer_pattern_p2'] = meta2.get('raw_bayer_pattern', '')
    all_metrics['gt_enabled'] = gt_ref is not None
    if gt_ref is not None:
        gt_desc = gt_ref.describe()
        all_metrics['gt_source'] = gt_desc.get('GT_Source', '')
        all_metrics['gt_subject'] = gt_desc.get('GT_Subject', '')
        all_metrics['gt_data_path'] = gt_desc.get('GT_Data_Path', '')
        all_metrics['gt_time_start_s'] = gt_desc.get('GT_Time_Start_s', np.nan)
        all_metrics['gt_time_end_s'] = gt_desc.get('GT_Time_End_s', np.nan)
        print(
            f"  GT参考: {all_metrics['gt_source']} subject={all_metrics['gt_subject']} "
            f"time={all_metrics['gt_time_start_s']:.2f}-{all_metrics['gt_time_end_s']:.2f}s"
        )

    signals = {}
    raw_multichannel = {}
    gt_window_rows = []
    for suffix, name, domain in [('_p1', name1, domain1), ('_p2', name2, domain2)]:
        signal, ch_name, signal_col = _pick_channel_from(merged, suffix, domain)
        print(f"  {name}: 使用通道 {ch_name} [{domain}]")
        all_metrics[f'{name}_Primary_Channel'] = signal_col
        signal = np.where(np.isnan(signal), 0.0, signal)

        # 保存原始信号用于 AC/DC 计算
        raw_signal = signal.copy()

        signal = preprocess(signal)
        signals[suffix] = signal
        ac = remove_dc(signal)
        ac_delta, ac_delta_col = _pick_ac_delta_from(merged, suffix, signal_col)
        all_metrics[f'{name}_Primary_AC_Delta'] = ac_delta_col

        td = time_domain_metrics(signal, ac, ac_delta, fps)
        for k, v in td.items():
            all_metrics[f'{name}_{k}'] = v

        fd, _, _ = freq_domain_metrics(signal, fps)
        for k, v in fd.items():
            all_metrics[f'{name}_{k}'] = v

        rr, _ = extract_rr_intervals(signal, fps)
        hm = hrv_metrics(rr, fps)
        for k, v in hm.items():
            all_metrics[f'{name}_{k}'] = v

        sq = sqi_metrics(signal, fps)
        for k, v in sq.items():
            all_metrics[f'{name}_{k}'] = v

        # 修正：使用原始信号计算 AC/DC 比
        dc = np.mean(raw_signal)
        ac_amplitude = np.std(raw_signal)
        all_metrics[f'{name}_AC_DC_Ratio'] = ac_amplitude / dc if dc > 1e-12 else 0

        if gt_ref is not None:
            gt_metrics, rows = gt_metrics_for_signal(
                signal,
                merged['Frame_ID'].values,
                fps,
                gt_ref,
                window_size_sec=gt_window_size,
                stride_sec=gt_stride,
                probe_time_mode=probe_time_mode,
                probe_time_offset_sec=probe_time_offset_sec,
                probe_name=name,
            )
            for k, v in gt_metrics.items():
                all_metrics[f'{name}_{k}'] = v
            gt_window_rows.extend(rows)

        if domain == 'rgb':
            c0 = merged[f'ROI_Mean_C0{suffix}'].values.astype(float)
            c1 = merged[f'ROI_Mean_C1{suffix}'].values.astype(float)
            c2 = merged[f'ROI_Mean_C2{suffix}'].values.astype(float)
            all_metrics[f'{name}_G_R_Ratio'] = np.nanmean(c1) / (np.nanmean(c0) + 1e-12)
            all_metrics[f'{name}_G_B_Ratio'] = np.nanmean(c1) / (np.nanmean(c2) + 1e-12)
            valid = ~(np.isnan(c0) | np.isnan(c1) | np.isnan(c2))
            if valid.sum() > 10:
                all_metrics[f'{name}_Channel_Corr_GR'], _ = pearsonr(c1[valid], c0[valid])
                all_metrics[f'{name}_Channel_Corr_GB'], _ = pearsonr(c1[valid], c2[valid])
        elif domain == 'raw':
            raw_multichannel[name] = _get_raw_multichannel_signals(merged, suffix)
            if raw_multichannel[name]:
                _store_raw_multichannel_metrics(all_metrics, name, raw_multichannel[name], fps)
                _store_raw_consistency_metrics(all_metrics, name, raw_multichannel[name], fps)

    # 对比指标：跨域时先归一化再比较（消除量纲差异）
    g1, g2 = signals['_p1'], signals['_p2']
    if is_cross_domain:
        g1_norm = (g1 - np.mean(g1)) / (np.std(g1) + 1e-12)
        g2_norm = (g2 - np.mean(g2)) / (np.std(g2) + 1e-12)
        cmp = comparison_metrics(g1_norm, g2_norm, fps)
        all_metrics['Compare_CrossDomain'] = True
    else:
        cmp = comparison_metrics(g1, g2, fps)
        all_metrics['Compare_CrossDomain'] = False
    for k, v in cmp.items():
        all_metrics[f'Compare_{k}'] = v
    if gt_ref is not None:
        _store_gt_delta_metrics(all_metrics, name1, name2)

    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    _build_summary_table(all_metrics, name1, name2).to_csv(summary_path, index=False)
    print(f"\n指标汇总已保存: {summary_path}")
    if gt_ref is not None:
        gt_window_path = os.path.join(output_dir, 'gt_window_metrics.csv')
        pd.DataFrame(gt_window_rows, columns=GT_WINDOW_METRIC_COLUMNS).to_csv(gt_window_path, index=False)
        print(f"GT窗口指标已保存: {gt_window_path}")

    print("\n========== 关键指标 ==========")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n生成可视化...")
    # 可视化：用 domain1 决定 p1 的通道，domain2 决定 p2 的通道
    _, ch_label1 = _domain_channel(domain1)
    _, ch_label2 = _domain_channel(domain2)
    ch_label = ch_label1 if not is_cross_domain else f"{ch_label1}/{ch_label2}"
    plot_timeseries(merged, fps, name1, name2, output_dir, domain1, domain2)
    plot_psd_cross(merged, fps, name1, name2, output_dir, domain1, domain2, ch_label)
    plot_spectrogram_cross(merged, fps, name1, name2, output_dir, domain1, domain2)
    for probe_name, raw_signals in raw_multichannel.items():
        if raw_signals:
            plot_raw_channel_timeseries(raw_signals, fps, probe_name, output_dir)
            plot_raw_channel_psd(raw_signals, fps, probe_name, output_dir)
            plot_raw_g1g2_overlay(raw_signals, fps, probe_name, output_dir)
            plot_raw_channel_correlation_heatmap(raw_signals, probe_name, output_dir)

    rr1, _ = extract_rr_intervals(g1, fps)
    rr2, _ = extract_rr_intervals(g2, fps)
    plot_poincare(rr1, rr2, name1, name2, output_dir)

    print(f"\n所有结果已保存到: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISP 探针 CSV 生理信号对比分析')
    parser.add_argument('csv1', help='探针1 CSV 路径')
    parser.add_argument('csv2', help='探针2 CSV 路径')
    parser.add_argument('--fps', type=float, default=30.0, help='视频帧率 (默认 30)')
    parser.add_argument('--output', default='probe_analysis/results', help='输出目录')
    parser.add_argument('--domain', default='auto', choices=['auto', 'raw', 'rgb', 'yuv'], help='域类型 (默认 auto)')
    parser.add_argument('--gt-source', default='none', choices=['none', CONTACT_PPG_SOURCE, 'raw_video_rppg'],
                        help='GT来源 (默认 none)')
    parser.add_argument('--gt-subject', default='', help='GT受试者名，例如 yjc/hyl/wyx')
    parser.add_argument('--gt-root', default=DEFAULT_GT_ROOT, help='contact_ppg GT根目录')
    parser.add_argument('--gt-path', default='', help='raw_video_rppg GT文件路径，需包含 bpm/times')
    parser.add_argument('--gt-window-size', type=float, default=GT_DEFAULT_WINDOW_SIZE, help='GT窗口长度秒 (默认 16)')
    parser.add_argument('--gt-stride', type=float, default=GT_DEFAULT_STRIDE, help='GT窗口步长秒 (默认 1)')
    parser.add_argument('--probe-time-mode', default='absolute_frame_id',
                        choices=['absolute_frame_id', 'relative_csv'],
                        help='探针时间轴对齐方式 (默认 absolute_frame_id)')
    parser.add_argument('--probe-time-offset-sec', type=float, default=0.0, help='探针时间轴额外偏移秒')
    args = parser.parse_args()
    gt_ref = None
    if args.gt_source != 'none':
        gt_ref = load_gt_reference(
            source_type=args.gt_source,
            subject=args.gt_subject,
            gt_root=args.gt_root,
            path=args.gt_path,
        )
    run_analysis(
        args.csv1,
        args.csv2,
        args.fps,
        args.output,
        args.domain,
        gt_ref=gt_ref,
        gt_window_size=args.gt_window_size,
        gt_stride=args.gt_stride,
        probe_time_mode=args.probe_time_mode,
        probe_time_offset_sec=args.probe_time_offset_sec,
    )
