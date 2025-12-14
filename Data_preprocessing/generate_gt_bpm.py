import numpy as np
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt
import json


"""
此模块用于从实验获取的bvps和times中直接处理获得rppg分析的bpms_gt和times_gt文件
"""

# --- 预处理函数 ---
def preprocess_ppg_simple(ppg_signal, fs):
    """
    更简洁的预处理,仅去趋势和标准化,
    把滤波任务交给频域的 spectrogram。
    """
    print("步骤 1/2: 正在执行去趋势 (Detrend)...")
    bvp_signal = signal.detrend(ppg_signal, type='linear')
    
    print("步骤 2/2: 正在执行标准化...")
    bvp_signal = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)
    
    return bvp_signal

# --- BVP处理器类 ---
class BVP_Processor:
    
    def __init__(self, data, fs, step_sec=1.0, nFFT=2048):
        """
        初始化处理器
        
        Parameters:
        -----------
        data : numpy array
            要处理的信号 (这里是 *预处理过的* bvp_sig)
        fs : float
            采样频率
        step_sec : float
            spectrogram的窗口步进时间 (秒)
        nFFT : int
            FFT点数
        """
        self.data = data
        self.fs = fs
        self.step = step_sec
        self.nFFT = nFFT
        
        # 初始化结果
        self.spect = None
        self.freqs = None
        self.times = None
        self.bpm = None

    def spectrogram(self, winsize=16):
        """
        使用 窗长winsize(秒)样本,计算仅限于 42-240 BPM 频段的 BVP 信号频谱图。
        """
        print("    正在计算 Spectrogram...")
        nperseg_int = int(self.fs * winsize)
        noverlap_int = int(self.fs * (winsize - self.step))

        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=nperseg_int,
                       noverlap=noverlap_int,
                       boundary='even',
                       nfft=self.nFFT)
        
        Z = np.squeeze(Z, axis=0)

        # freq subband (0.65 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        
        self.spect = np.abs(Z[band, :])
        self.freqs = 60*F[band]
        self.times = T
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        """
        print("  调用 getBPM...")
        self.spectrogram(winsize)
        return self.bpm, self.times

# --- GT文件保存函数 ---
def save_gt_file(bpm_values, times, output_path, metadata=None):
    """
    保存Ground Truth BPM数据到文件
    
    Parameters:
    -----------
    bpm_values : numpy array
        BPM值数组
    times : numpy array
        时间戳数组
    output_path : str
        输出文件路径(不含扩展名)
    metadata : dict
        额外的元数据信息
    """
    # 保存为.npz格式
    np.savez(f"{output_path}.npz", 
             bpm=bpm_values, 
             times=times,
             metadata=metadata if metadata else {})
    
    # 同时保存为JSON格式
    gt_data = {
        'bpm': bpm_values.tolist(),
        'times': times.tolist(),
        'metadata': metadata if metadata else {}
    }
    with open(f"{output_path}.json", 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ GT文件已保存:")
    print(f"  - {output_path}.npz")
    print(f"  - {output_path}.json")
    print(f"{'='*60}")

# --- 主程序 ---
if __name__ == "__main__":
    
    # ========== 配置区域 - 请修改这里的路径 ==========
    ppg_file_path = 'Data_for_pyVHR/gt_data/gt_wzx_nAE/HK-PPG-COM7_sig.npy'
    ts_file_path = 'Data_for_pyVHR/gt_data/gt_wzx_nAE/HK-PPG-COM7_ts.npy'
    gt_output_path = 'Data_for_pyVHR/gt_data/gt_wzx_nAE/bpms_times_GT'
    
    # 处理参数
    window_sec = 16
    BPM_target = 0.5   # 目标BPM分辨率
    # ===============================================
    
    print(f"\n{'='*60}")
    print(f"开始生成Ground Truth BPM文件")
    print(f"{'='*60}")
    
    # 1. 加载原始数据
    print(f"\n步骤 1: 加载原始PPG信号和时间戳...")
    try:
        raw_ppg_signal_1d = np.load(ppg_file_path)
        raw_ppg_signal = raw_ppg_signal_1d.reshape(1, -1)
        ts_signal = np.load(ts_file_path)
        print(f"  ✓ 成功加载 {len(raw_ppg_signal_1d)} 个数据点")
    except FileNotFoundError as e:
        print(f"  ✗ 错误: 找不到文件")
        print(f"    {e}")
        exit()

    # 2. 计算采样率
    print(f"\n步骤 2: 计算采样率...")
    fs = 1.0 / np.mean(np.diff(ts_signal))
    print(f"  ✓ 动态计算采样率 Fs: {fs:.2f} Hz")

    # 3. 计算nFFT
    print(f"\n步骤 3: 计算FFT点数...")
    nFFT_need = (fs * 60) / BPM_target
    nFFT_normalized = int(2**np.ceil(np.log2(nFFT_need)))
    print(f"  ✓ 规范化后的 nFFT (2的幂): {nFFT_normalized}")

    # 4. 执行预处理
    print(f"\n步骤 4: 执行预处理...")
    bvp_sig_simple = preprocess_ppg_simple(raw_ppg_signal, fs)
    print(f"  ✓ 预处理完成")

    # 5. 计算BPM
    print(f"\n步骤 5: 计算BPM...")
    processor_simple = BVP_Processor(data=bvp_sig_simple, fs=fs, 
                                    step_sec=1.0, nFFT=nFFT_normalized)
    bpm_simple, times_simple = processor_simple.getBPM(winsize=window_sec)
    print(f"  ✓ 计算完成！共得到 {len(bpm_simple)} 个BPM值")

    # 6. 打印结果摘要
    print(f"\n{'='*60}")
    print(f"结果摘要:")
    print(f"{'='*60}")
    print(f"BPM数量: {len(bpm_simple)}")
    print(f"时间范围: {times_simple[0]:.2f}s - {times_simple[-1]:.2f}s")
    print(f"BPM范围: {np.min(bpm_simple):.2f} - {np.max(bpm_simple):.2f}")
    print(f"平均BPM: {np.mean(bpm_simple):.2f}")
    print(f"\n前5个BPM值: {bpm_simple[:5]}")
    print(f"对应时间: {times_simple[:5]}")
    
    # 7. 保存GT文件
    print(f"\n步骤 6: 保存GT文件...")
    metadata = {
        'source_ppg_file': ppg_file_path,
        'source_ts_file': ts_file_path,
        'sampling_rate': float(fs),
        'window_size': window_sec,
        'nFFT': nFFT_normalized,
        'bpm_resolution': BPM_target,
        'num_samples': int(len(bpm_simple)),
        'time_range': [float(times_simple[0]), float(times_simple[-1])],
        'bpm_range': [float(np.min(bpm_simple)), float(np.max(bpm_simple))],
        'mean_bpm': float(np.mean(bpm_simple))
    }
    
    save_gt_file(bpm_simple, times_simple, gt_output_path, metadata)
    
    print(f"\n✓ 全部完成！GT文件已生成。")
    print(f"  下一步: 在 analyze_with_pyvhr.py 中使用该GT文件进行误差分析")