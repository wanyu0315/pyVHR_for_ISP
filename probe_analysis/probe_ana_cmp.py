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
_cjk_ttf = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
_zh_prop = None
if os.path.isfile(_cjk_ttf):
    fm.fontManager.addfont(_cjk_ttf)
    _zh_prop = fm.FontProperties(fname=_cjk_ttf)
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 模块 0: 数据加载与预处理
# ============================================================

def detect_domain(csv_path: str, df: pd.DataFrame) -> str:
    name = os.path.basename(os.path.dirname(csv_path)).lower()
    # 探针名格式: "前一模块-后一模块"，域由前一模块决定
    # YUV域：前一模块是 colorspace/contrastsaturation/yuvtorgb
    yuv_prefixes = ['colorspace-', 'contrastsaturation-', 'yuvtorgb-']
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
    merged = pd.merge(df1, df2, on='Frame_ID', suffixes=('_p1', '_p2'))
    merged = merged.sort_values('Frame_ID').reset_index(drop=True)
    merged = merged.interpolate(method='linear', limit_direction='both')
    merged = merged.dropna(subset=['Frame_ID'])
    return merged


def bandpass_filter(x: np.ndarray, fps: float, lo: float = 0.7, hi: float = 2.0, order: int = 4):
    """Butterworth 带通滤波"""
    if len(x) < 30:
        return np.zeros_like(x)
    nyq = fps / 2.0
    if hi >= nyq:
        hi = nyq - 0.01
    b, a = sig.butter(order, [lo / nyq, hi / nyq], btype='band')
    return sig.filtfilt(b, a, x)


def remove_dc(x: np.ndarray):
    return x - np.mean(x)


def preprocess(x: np.ndarray):
    """预处理：中值滤波去尖刺 + 线性去趋势"""
    x = median_filter(x, size=3)
    x = sig.detrend(x, type='linear')
    return x + np.mean(x)  # 保留 DC 偏置，只去除漂移

# ============================================================
# 模块 1: 时域分析
# ============================================================

def time_domain_metrics(raw: np.ndarray, ac: np.ndarray, ac_delta: np.ndarray, fps: float):
    # SNR: HR band power vs out-of-band power (from PSD)
    nperseg = min(256, max(16, len(raw) // 4))
    freqs, psd = sig.welch(remove_dc(raw), fs=fps, nperseg=nperseg)
    hr_mask = (freqs >= 0.7) & (freqs <= 2.0)
    hr_power = np.trapz(psd[hr_mask], freqs[hr_mask]) if np.any(hr_mask) else 0
    noise_mask = ~hr_mask & (freqs > 0)
    noise_power = np.trapz(psd[noise_mask], freqs[noise_mask]) if np.any(noise_mask) else 0
    snr = 10 * np.log10(hr_power / noise_power) if noise_power > 0 else np.inf

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

    hr_mask = (freqs >= 0.7) & (freqs <= 2.0)
    hr_power = np.trapz(psd[hr_mask], freqs[hr_mask])
    total_power = np.trapz(psd, freqs)
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
    min_dist = int(fps * 0.4)
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
    lf = np.trapz(psd_rr[lf_mask], f_rr[lf_mask])
    hf = np.trapz(psd_rr[hf_mask], f_rr[hf_mask])
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

    peaks, props = sig.find_peaks(bp, distance=int(fps * 0.4), prominence=np.std(bp) * 0.1)
    avg_prominence = np.mean(props['prominences']) if len(peaks) > 0 else 0
    prominence_ratio = avg_prominence / (np.std(bp) + 1e-12)

    ac_corr = np.correlate(bp, bp, mode='full')
    ac_corr = ac_corr[len(bp) - 1:]
    ac_corr /= ac_corr[0] + 1e-12
    min_lag = int(fps / 2.0)
    max_lag = int(fps / 0.7)
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
    dtw_dist = np.sqrt(np.mean((n1 - n2) ** 2))

    def _est_bpm(x):
        nperseg = min(256, max(16, len(x) // 4))
        f, p = sig.welch(remove_dc(x), fs=fps, nperseg=nperseg)
        mask = (f >= 0.7) & (f <= 2.0)
        return f[mask][np.argmax(p[mask])] * 60 if np.any(mask) else 0
    bpm_diff = _est_bpm(raw2) - _est_bpm(raw1)

    def _snr(x):
        nperseg = min(256, max(16, len(x) // 4))
        f, p = sig.welch(remove_dc(x), fs=fps, nperseg=nperseg)
        hr = (f >= 0.7) & (f <= 2.0)
        noise = ~hr & (f > 0)
        hp = np.trapz(p[hr], f[hr]) if np.any(hr) else 0
        np_ = np.trapz(p[noise], f[noise]) if np.any(noise) else 0
        return 10 * np.log10(hp / np_) if np_ > 0 else 0
    delta_snr = _snr(raw2) - _snr(raw1)

    nperseg = min(256, max(16, len(raw1) // 4))
    _, psd1 = sig.welch(remove_dc(raw1), fs=fps, nperseg=nperseg)
    _, psd2 = sig.welch(remove_dc(raw2), fs=fps, nperseg=nperseg)
    spectral_cosine = 1.0 - cosine_dist(psd1, psd2)

    return {
        'PCC': pcc,
        'Phase_Delay_ms': phase_delay_ms,
        'DTW_Distance': dtw_dist,
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

        ax.set_ylabel(f'AC 幅度', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    domain_str = f'{domain1.upper()}→{domain2.upper()}' if domain1 != domain2 else domain1.upper()
    fig.suptitle(f'ROI 通道时域波形对比 (Bandpass 0.7-2.0 Hz) [{domain_str}]', fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'timeseries_comparison.png'), dpi=150)
    plt.close(fig)


def _get_signal_for_plot(merged: pd.DataFrame, suffix: str, domain: str) -> np.ndarray:
    col_idx, _ = _domain_channel(domain)
    raw = merged[f'ROI_Mean_C{col_idx}{suffix}'].values.astype(float)
    if np.all(np.isnan(raw)):
        raw = merged[f'ROI_Mean_C0{suffix}'].values.astype(float)
    raw = np.where(np.isnan(raw), np.nanmean(raw) if not np.all(np.isnan(raw)) else 0, raw)
    return preprocess(raw)


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

        mask_4hz = f <= 4.0
        f_plot = f[mask_4hz]
        psd_plot = psd[mask_4hz]

        hr_mask = (f >= 0.7) & (f <= 2.0)
        if np.any(hr_mask) and np.max(psd[hr_mask]) > 0:
            peak_idx = np.argmax(psd[hr_mask])
            peak_freq = f[hr_mask][peak_idx]
            peak_bpm = peak_freq * 60
            peak_psd = psd[hr_mask][peak_idx]

            peak_band = (f >= peak_freq - 0.1) & (f <= peak_freq + 0.1)
            signal_power = np.trapz(psd[peak_band], f[peak_band])
            noise_mask = (f > 0) & (f <= 4.0) & ~peak_band
            noise_power = np.trapz(psd[noise_mask], f[noise_mask])
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-12 else 0

            snr_data[suffix] = snr_db
            peak_data[suffix] = (peak_freq, peak_bpm, peak_psd)
        else:
            snr_data[suffix] = 0
            peak_data[suffix] = (0, 0, 0)

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

    ax.axvspan(0.7, 2.0, alpha=0.15, color='red', label='HR Band')

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
    ax.set_title(f'PSD 对比 [{ch_label}]', fontsize=13, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 4.0)
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
        ax.axhline(0.7, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(2.0, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
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
        return 0, 'C0(RAW/Luma)'
    elif domain == 'yuv':
        return 0, 'C0(Y/Luma)'
    else:
        return 1, 'C1(G)'


def _pick_channel_from(merged: pd.DataFrame, suffix: str, domain: str):
    col_idx, label = _domain_channel(domain)
    vals = merged[f'ROI_Mean_C{col_idx}{suffix}']
    if vals.notna().sum() > 10:
        return vals.values, label
    for idx in [0, 1, 2]:
        c = merged[f'ROI_Mean_C{idx}{suffix}']
        if c.notna().sum() > 10:
            return c.values, f'C{idx}(fallback)'
    return merged[f'ROI_Mean_C0{suffix}'].values, 'C0(fallback)'


def run_analysis(csv1: str, csv2: str, fps: float, output_dir: str, domain_arg: str = 'auto'):
    name1 = os.path.splitext(os.path.basename(csv1))[0]
    name2 = os.path.splitext(os.path.basename(csv2))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"加载数据: {csv1} vs {csv2}")
    merged = load_and_align(csv1, csv2)
    print(f"对齐后帧数: {len(merged)}")
    if len(merged) == 0:
        print("错误：对齐后无有效帧，请检查两个CSV的Frame_ID是否有交集。")
        return

    # 每个探针独立检测域
    if domain_arg == 'auto':
        domain1 = detect_domain(csv1, pd.read_csv(csv1))
        domain2 = detect_domain(csv2, pd.read_csv(csv2))
    else:
        domain1 = domain2 = domain_arg

    is_cross_domain = (domain1 != domain2)
    domain_display = f"{domain1.upper()}→{domain2.upper()}" if is_cross_domain else domain1.upper()
    print(f"  探针1域: {domain1}  探针2域: {domain2}" + (" [跨域对比]" if is_cross_domain else ""))

    all_metrics = {}
    all_metrics['domain_p1'] = domain1
    all_metrics['domain_p2'] = domain2

    signals = {}
    for suffix, name, domain in [('_p1', name1, domain1), ('_p2', name2, domain2)]:
        signal, ch_name = _pick_channel_from(merged, suffix, domain)
        print(f"  {name}: 使用通道 {ch_name} [{domain}]")
        signal = np.where(np.isnan(signal), 0.0, signal)
        signal = preprocess(signal)
        signals[suffix] = signal
        ac = remove_dc(signal)
        ac_delta = merged[f'ROI_AC_Delta{suffix}'].values

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

        all_metrics[f'{name}_AC_DC_Ratio'] = np.std(signal) / (np.mean(signal) + 1e-12)

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

    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    pd.DataFrame(all_metrics.items(), columns=['metric', 'value']).to_csv(summary_path, index=False)
    print(f"\n指标汇总已保存: {summary_path}")

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
    args = parser.parse_args()
    run_analysis(args.csv1, args.csv2, args.fps, args.output, args.domain)
