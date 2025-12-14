import numpy as np
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt


#  # 1. 标准数据集GT文件输入
# input_txt_file = 'Data_for_pyVHR/UBFC2/subject3/ground_truth.txt'    # 替换成你的 .txt 文件名
# output_npy_file = 'Data_preprocessing/ubfc2_test.npy'  # 你想要输出的 .npy 文件名

# try:
#     print(f"正在读取文件: {input_txt_file}...")
    
#     # --- 2. 打开文件并只读取第一行 ---
#     with open(input_txt_file, 'r') as f:
#         first_line_str = f.readline()

#     # --- 3. 将字符串处理并转换为 NumPy 数组 ---
    
#     # .split() - 默认按所有空白(空格, Tab等)拆分成一个字符串列表
#     string_list = first_line_str.strip().split()
    
#     # 将字符串列表转换为 float 类型的 NumPy 数组
#     data_array = np.array(string_list, dtype=float)

#     print(f"成功读取并转换第一行:")
#     print(data_array)
#     print(f"数组形状: {data_array.shape}")

#     # --- 4. 将数组保存为 .npy 文件 ---
#     np.save(output_npy_file, data_array)

#     print(f"\n[成功] 数组已保存到: {output_npy_file}")
    

# except FileNotFoundError:
#     print(f"错误: 找不到文件 {input_txt_file}")
# except ValueError:
#     print(f"错误: 第一行 '{first_line_str.strip()}' 包含无法转换为数字的字符。")
# except Exception as e:
#     print(f"发生未知错误: {e}")

"""
旨在测试从gt文件中得到bpm_gt
"""


# --- 推荐的预处理函数 (只去趋势和标准化) ---
def preprocess_ppg_simple(ppg_signal, fs):
    """
    更简洁的预处理，仅去趋势和标准化，
    把滤波任务交给频域的 spectrogram。
    """
    print("步骤 1/2: 正在执行去趋势 (Detrend)...")
    bvp_signal = signal.detrend(ppg_signal, type='linear')

    # print("步骤 1/2: 正在执行时域带通滤波 (0.65-4.0 Hz)...")
    
    # # 注意：这里的频率范围与 spectrogram 中的一致
    # low_hz = 0.65
    # high_hz = 4.0
    
    # # 使用二阶 Butterworth 滤波器
    # sos = signal.butter(2, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos')
    
    # # 使用 filtfilt 进行零相位滤波
    # bvp_signal = signal.sosfilt(sos, ppg_signal)
    
    print("步骤 2/2: 正在执行标准化...")
    bvp_signal = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)
    
    return bvp_signal

# --- 步骤 2: 封装一个类来容纳方法 ---
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
        self.step = step_sec  # getBPM/spectrogram 需要这个
        self.nFFT = nFFT      # spectrogram 需要这个
        
        # 初始化结果
        self.spect = None
        self.freqs = None
        self.times = None
        self.bpm = None

    def spectrogram(self, winsize=5):
        """
        使用 winsize（秒）样本，计算仅限于 42-240 BPM 频段的 BVP 信号频谱图。
        (这是 pyVHR 源码)
        """
        print("    正在计算 Spectrogram...")
        # 确保nperseg/noverlap是整数
        nperseg_int = int(self.fs * winsize)
        noverlap_int = int(self.fs * (winsize - self.step))

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=nperseg_int,
                       noverlap=noverlap_int,
                       boundary='even',
                       nfft=self.nFFT)
        
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.65 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        
        self.spect = np.abs(Z[band, :])   # spectrum magnitude
        self.freqs = 60*F[band]           # spectrum freq in bpm
        self.times = T                    # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        (pyVHR 源码)
        """
        print("  调用 getBPM...")
        self.spectrogram(winsize)
        return self.bpm, self.times

# --- 推荐的执行流程 ---
if __name__ == "__main__":
    
    # 定义输入
    ppg_file_path = 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_sig.npy'
    ts_file_path = 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_ts.npy'
    
    print(f"\n--- 执行'预处理'流程 ---")
    
    # 1. 加载
    try:
        raw_ppg_signal_1d = np.load(ppg_file_path)
        raw_ppg_signal = raw_ppg_signal_1d.reshape(1, -1) # 重塑为2D，兼容pyVHR源码的getBPM函数输入
        ts_signal = np.load(ts_file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {ppg_file_path}")
        exit()

    # 计算采样率
    fs = 1.0 / np.mean(np.diff(ts_signal)) # <--  动态计算 Fs
    print(f"动态计算采样率 Fs: {fs:.2f} Hz")

    # 计算nFFT
    BPM_targe = 0.5                     # 目标BPM分辨率
    nFFT_need = (fs * 60)/BPM_targe     # 至少所需FFT点数
    nFFT_normalized = int(2**np.ceil(np.log2(nFFT_need)))    # 规范化
    print(f"规范化后的 nFFT (2的幂): {nFFT_normalized}")

    # 2. 执行预处理函数
    bvp_sig_simple = preprocess_ppg_simple(raw_ppg_signal, fs)


    # 3. 实例化处理器 (使用简洁处理后的信号)
    processor_simple = BVP_Processor(data=bvp_sig_simple, fs=fs, step_sec=1.0, nFFT=nFFT_normalized)
    
    # 4. 调用 getBPM 方法
    window_sec = 8
    bpm_simple, times_simple = processor_simple.getBPM(winsize=window_sec)


    # 5. 打印结果
    print("\n--- 结果 (简洁方案) ---")
    print(f"计算完成！共得到 {len(bpm_simple)} 个BPM值。")
    print(f"时间 (秒): {times_simple}")
    print(f"BPM: {bpm_simple}")
    
    # 6. 保存和绘图
    np.save('Data_preprocessing/bpm_from_simple.npy', bpm_simple)
    plt.figure(figsize=(15, 5))
    plt.plot(times_simple, bpm_simple, 'b-o', label=f'Calculated BPM (Win={window_sec}s)')
    plt.title("BPM Signal (from Simple Pre-processing)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Heart Rate (BPM)")
    plt.grid(True)
    plt.legend()
    plt.savefig("Data_preprocessing/bpm_plot_from_simple.png")
    print("BPM 结果已保存为 'bpm_from_simple.npy' 和 'bpm_plot_from_simple.png'")