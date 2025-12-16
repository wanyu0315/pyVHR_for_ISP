# black_level_calibration.py
"""
黑电平标定工具
用于从暗场视频帧中测量传感器的黑电平
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class BlackLevelCalibrator:
    """
    黑电平标定器
    从全黑视频帧中分析和计算黑电平
    """
    
    def __init__(self):
        """初始化标定器"""
        self.dark_frames = []
        self.black_levels = {}
    
    def load_dark_frames(self, dark_frame_folder: str, 
                        width: int, height: int, 
                        dtype: type = np.uint16,
                        pattern: str = '*.raw') -> int:
        """
        加载暗场视频帧
        
        Args:
            dark_frame_folder: 暗场帧文件夹路径
            width: 图像宽度
            height: 图像高度
            dtype: 数据类型（np.uint16 或 np.uint8）
            pattern: 文件匹配模式
        
        Returns:
            加载的帧数
        """
        print("\n" + "="*70)
        print("加载暗场帧")
        print("="*70)
        
        raw_files = sorted(glob.glob(os.path.join(dark_frame_folder, pattern)))
        
        if not raw_files:
            print(f"错误：在 '{dark_frame_folder}' 中未找到 {pattern} 文件")
            return 0
        
        print(f"找到 {len(raw_files)} 个暗场帧文件")
        
        self.dark_frames = []
        for i, raw_file in enumerate(raw_files):
            try:
                # 读取无头RAW文件
                with open(raw_file, 'rb') as f:
                    raw_data = np.fromfile(f, dtype=dtype)
                
                # 重塑为2D图像
                if raw_data.size == width * height:
                    frame = raw_data.reshape(height, width)
                    self.dark_frames.append(frame)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  已加载: {i+1}/{len(raw_files)} 帧")
                else:
                    print(f"  警告: {os.path.basename(raw_file)} 大小不匹配，跳过")
            
            except Exception as e:
                print(f"  错误: 无法加载 {os.path.basename(raw_file)}: {e}")
                continue
        
        print(f"✓ 成功加载 {len(self.dark_frames)} 帧")
        return len(self.dark_frames)
    
    def calibrate(self, bayer_pattern: str = 'GBRG',
                 method: str = 'robust') -> Dict[str, float]:
        """
        执行黑电平标定
        
        Args:
            bayer_pattern: Bayer模式 ('RGGB', 'BGGR', 'GRBG', 'GBRG')
            method: 标定方法
                - 'mean': 简单平均（快速但对异常值敏感）
                - 'median': 中位数（更robust）
                - 'robust': Robust统计（推荐）
                - 'percentile': 百分位数方法
        
        Returns:
            黑电平字典 {'R': value, 'Gr': value, 'Gb': value, 'B': value, 'overall': value}
        """
        if not self.dark_frames:
            raise ValueError("请先加载暗场帧数据")
        
        print("\n" + "="*70)
        print(f"黑电平标定 - 方法: {method}, Bayer模式: {bayer_pattern}")
        print("="*70)
        
        # 1. 将所有帧堆叠成3D数组 (frames, height, width)
        dark_stack = np.stack(self.dark_frames, axis=0)
        print(f"暗场数据形状: {dark_stack.shape}")
        print(f"数据类型: {dark_stack.dtype}")
        print(f"数据范围: [{dark_stack.min()}, {dark_stack.max()}]")
        
        # 2. 分离Bayer通道
        channels = self._extract_bayer_channels(dark_stack, bayer_pattern)
        
        # 3. 对每个通道计算黑电平
        self.black_levels = {}
        
        print("\n各通道黑电平分析:")
        print("-" * 70)
        
        for color in ['R', 'Gr', 'Gb', 'B']:
            channel_data = channels[color]
            
            if method == 'mean':
                black_level = np.mean(channel_data)
            elif method == 'median':
                black_level = np.median(channel_data)
            elif method == 'robust':
                black_level = self._robust_estimator(channel_data)
            elif method == 'percentile':
                black_level = np.percentile(channel_data, 50)  # 50分位 = 中位数
            else:
                raise ValueError(f"未知的方法: {method}")
            
            self.black_levels[color] = float(black_level)
            
            # 打印详细统计
            print(f"\n{color} 通道:")
            print(f"  平均值:    {np.mean(channel_data):.2f}")
            print(f"  中位数:    {np.median(channel_data):.2f}")
            print(f"  标准差:    {np.std(channel_data):.2f}")
            print(f"  最小值:    {np.min(channel_data):.2f}")
            print(f"  最大值:    {np.max(channel_data):.2f}")
            print(f"  → 黑电平:  {black_level:.2f}")
        
        # 4. 计算整体黑电平（四个通道的平均）
        self.black_levels['overall'] = np.mean([
            self.black_levels['R'],
            self.black_levels['Gr'],
            self.black_levels['Gb'],
            self.black_levels['B']
        ])
        
        # 5. 计算绿色通道的平均（Gr + Gb）
        self.black_levels['G'] = (self.black_levels['Gr'] + self.black_levels['Gb']) / 2.0
        
        print("\n" + "="*70)
        print("标定结果汇总:")
        print("="*70)
        print(f"R  通道黑电平: {self.black_levels['R']:.2f}")
        print(f"Gr 通道黑电平: {self.black_levels['Gr']:.2f}")
        print(f"Gb 通道黑电平: {self.black_levels['Gb']:.2f}")
        print(f"B  通道黑电平: {self.black_levels['B']:.2f}")
        print(f"G  平均黑电平: {self.black_levels['G']:.2f}")
        print(f"整体黑电平:    {self.black_levels['overall']:.2f}")
        print("="*70)
        
        return self.black_levels
    
    def _extract_bayer_channels(self, dark_stack: np.ndarray, 
                                bayer_pattern: str) -> Dict[str, np.ndarray]:
        """
        从Bayer图像堆栈中提取R/Gr/Gb/B四个通道
        
        Returns:
            {'R': array, 'Gr': array, 'Gb': array, 'B': array}
        """
        patterns = {
            'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
            'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
            'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
            'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)}
        }
        
        if bayer_pattern not in patterns:
            raise ValueError(f"不支持的Bayer模式: {bayer_pattern}")
        
        layout = patterns[bayer_pattern]
        channels = {}
        
        for color, (row_offset, col_offset) in layout.items():
            # 提取所有帧的该通道数据
            channel_data = dark_stack[:, row_offset::2, col_offset::2]
            channels[color] = channel_data.flatten()
        
        return channels
    
    def _robust_estimator(self, data: np.ndarray) -> float:
        """
        Robust黑电平估计
        使用截断平均（去除极端值后的平均）
        """
        # 去除最低和最高的5%数据
        lower = np.percentile(data, 5)
        upper = np.percentile(data, 95)
        
        trimmed_data = data[(data >= lower) & (data <= upper)]
        
        return np.mean(trimmed_data)
    
    def analyze_temporal_stability(self) -> Dict[str, np.ndarray]:
        """
        分析黑电平的时间稳定性
        检查黑电平是否随时间漂移
        
        Returns:
            每帧的平均值序列
        """
        if not self.dark_frames:
            raise ValueError("请先加载暗场帧数据")
        
        print("\n" + "="*70)
        print("时间稳定性分析")
        print("="*70)
        
        frame_means = np.array([np.mean(frame) for frame in self.dark_frames])
        frame_stds = np.array([np.std(frame) for frame in self.dark_frames])
        
        print(f"平均值变化范围: [{frame_means.min():.2f}, {frame_means.max():.2f}]")
        print(f"平均值标准差:   {np.std(frame_means):.2f}")
        print(f"最大帧间差异:   {np.max(np.abs(np.diff(frame_means))):.2f}")
        
        if np.std(frame_means) > 5:
            print("⚠️  警告: 黑电平时间不稳定，可能受温度影响")
        else:
            print("✓  黑电平时间稳定")
        
        return {
            'frame_means': frame_means,
            'frame_stds': frame_stds
        }
    
    def analyze_spatial_uniformity(self, bayer_pattern: str = 'GBRG') -> Dict:
        """
        分析黑电平的空间均匀性
        检查图像不同区域的黑电平是否一致
        
        Returns:
            空间分析结果
        """
        if not self.dark_frames:
            raise ValueError("请先加载暗场帧数据")
        
        print("\n" + "="*70)
        print("空间均匀性分析")
        print("="*70)
        
        # 使用所有帧的平均作为代表
        avg_dark_frame = np.mean(self.dark_frames, axis=0)
        
        h, w = avg_dark_frame.shape
        
        # 将图像分成9个区域（3×3网格）
        regions = {}
        region_names = [
            'top_left', 'top_center', 'top_right',
            'mid_left', 'center', 'mid_right',
            'bot_left', 'bot_center', 'bot_right'
        ]
        
        for i, row in enumerate(['top', 'mid', 'bot']):
            for j, col in enumerate(['left', 'center', 'right']):
                y_start = i * h // 3
                y_end = (i + 1) * h // 3
                x_start = j * w // 3
                x_end = (j + 1) * w // 3
                
                region = avg_dark_frame[y_start:y_end, x_start:x_end]
                region_name = f"{row}_{col}"
                regions[region_name] = {
                    'mean': np.mean(region),
                    'std': np.std(region)
                }
        
        # 打印结果
        print("\n区域黑电平 (平均值):")
        print(f"  左上: {regions['top_left']['mean']:.2f}    "
              f"中上: {regions['top_center']['mean']:.2f}    "
              f"右上: {regions['top_right']['mean']:.2f}")
        print(f"  左中: {regions['mid_left']['mean']:.2f}    "
              f"中心: {regions['mid_center']['mean']:.2f}    "
              f"右中: {regions['mid_right']['mean']:.2f}")
        print(f"  左下: {regions['bot_left']['mean']:.2f}    "
              f"中下: {regions['bot_center']['mean']:.2f}    "
              f"右下: {regions['bot_right']['mean']:.2f}")
        
        # 计算均匀性指标
        region_means = [r['mean'] for r in regions.values()]
        uniformity = np.std(region_means)
        
        print(f"\n空间均匀性 (区域间标准差): {uniformity:.2f}")
        
        if uniformity > 10:
            print("⚠️  警告: 黑电平空间不均匀，可能需要逐像素校正")
        else:
            print("✓  黑电平空间均匀")
        
        return {
            'regions': regions,
            'uniformity': uniformity,
            'avg_frame': avg_dark_frame
        }
    
    def generate_report(self, output_folder: str = 'black_level_calibration'):
        """
        生成完整的标定报告（包括可视化）
        
        Args:
            output_folder: 输出文件夹
        """
        if not self.dark_frames or not self.black_levels:
            raise ValueError("请先执行标定")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        print("\n" + "="*70)
        print("生成标定报告")
        print("="*70)
        
        # 1. 保存黑电平数值
        report_path = os.path.join(output_folder, 'black_level_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("黑电平标定报告\n")
            f.write("="*70 + "\n\n")
            
            f.write("## 标定结果\n\n")
            for channel, value in self.black_levels.items():
                f.write(f"{channel:8s}: {value:.2f}\n")
            
            f.write("\n## 使用建议\n\n")
            f.write("在ISP流程中使用以下参数:\n\n")
            f.write("```python\n")
            f.write("processing_params = {\n")
            f.write("    'blacklevelcorrection': {\n")
            f.write(f"        'black_level': {int(self.black_levels['overall'])},\n")
            f.write("        # 或者分通道校正:\n")
            f.write(f"        # 'R': {int(self.black_levels['R'])},\n")
            f.write(f"        # 'G': {int(self.black_levels['G'])},\n")
            f.write(f"        # 'B': {int(self.black_levels['B'])},\n")
            f.write("    }\n")
            f.write("}\n")
            f.write("```\n")
        
        print(f"✓ 报告已保存: {report_path}")
        
        # 2. 生成可视化图表
        self._plot_analysis(output_folder)
        
        print(f"✓ 可视化已保存到: {output_folder}")
    
    def _plot_analysis(self, output_folder: str):
        """生成分析图表"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 黑电平柱状图
        ax1 = plt.subplot(2, 3, 1)
        channels = ['R', 'Gr', 'Gb', 'B']
        values = [self.black_levels[ch] for ch in channels]
        colors = ['red', 'green', 'lightgreen', 'blue']
        
        bars = ax1.bar(channels, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('黑电平值', fontsize=12)
        ax1.set_title('各通道黑电平', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 2. 暗场帧平均图像
        ax2 = plt.subplot(2, 3, 2)
        avg_frame = np.mean(self.dark_frames, axis=0)
        im2 = ax2.imshow(avg_frame, cmap='gray', vmin=avg_frame.min(), vmax=avg_frame.max())
        ax2.set_title('平均暗场图像', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='像素值')
        
        # 3. 暗场帧标准差图像
        ax3 = plt.subplot(2, 3, 3)
        std_frame = np.std(self.dark_frames, axis=0)
        im3 = ax3.imshow(std_frame, cmap='hot', vmin=0, vmax=np.percentile(std_frame, 99))
        ax3.set_title('暗场标准差图（噪声分布）', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='标准差')
        
        # 4. 时间稳定性
        if len(self.dark_frames) > 1:
            ax4 = plt.subplot(2, 3, 4)
            frame_means = [np.mean(frame) for frame in self.dark_frames]
            ax4.plot(frame_means, 'o-', linewidth=2, markersize=4)
            ax4.set_xlabel('帧序号', fontsize=12)
            ax4.set_ylabel('平均黑电平', fontsize=12)
            ax4.set_title('时间稳定性分析', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        # 5. 直方图 - 整体分布
        ax5 = plt.subplot(2, 3, 5)
        all_pixels = np.concatenate([frame.flatten() for frame in self.dark_frames])
        ax5.hist(all_pixels, bins=100, color='gray', alpha=0.7, edgecolor='black')
        ax5.axvline(self.black_levels['overall'], color='red', linestyle='--', 
                   linewidth=2, label=f"整体黑电平: {self.black_levels['overall']:.1f}")
        ax5.set_xlabel('像素值', fontsize=12)
        ax5.set_ylabel('频数', fontsize=12)
        ax5.set_title('黑电平分布直方图', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. 各通道直方图对比
        ax6 = plt.subplot(2, 3, 6)
        dark_stack = np.stack(self.dark_frames, axis=0)
        
        for color, hist_color in zip(['R', 'Gr', 'Gb', 'B'], 
                                     ['red', 'green', 'lightgreen', 'blue']):
            if color == 'R':
                data = dark_stack[:, 1::2, 0::2].flatten()  # 示例位置，需根据实际Bayer调整
            elif color == 'Gr':
                data = dark_stack[:, 0::2, 1::2].flatten()
            elif color == 'Gb':
                data = dark_stack[:, 1::2, 1::2].flatten()
            else:  # B
                data = dark_stack[:, 0::2, 0::2].flatten()
            
            ax6.hist(data, bins=50, alpha=0.5, label=color, color=hist_color, edgecolor='black')
        
        ax6.set_xlabel('像素值', fontsize=12)
        ax6.set_ylabel('频数', fontsize=12)
        ax6.set_title('各通道黑电平分布', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_folder, 'black_level_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


# ===================================================================
# 使用示例
# ===================================================================

def calibrate_black_level(dark_frame_folder: str,
                         output_folder: str,
                         width: int = 1280,
                         height: int = 800,
                         bayer_pattern: str = 'GBRG',
                         dtype: type = np.uint16):
    """
    完整的黑电平标定流程
    
    Args:
        dark_frame_folder: 全黑视频帧文件夹
        width: 图像宽度
        height: 图像高度
        bayer_pattern: Bayer模式
        dtype: 数据类型
    """
    # 创建标定器
    calibrator = BlackLevelCalibrator()
    
    # 1. 加载暗场帧
    num_frames = calibrator.load_dark_frames(
        dark_frame_folder, 
        width, 
        height, 
        dtype
    )
    
    if num_frames == 0:
        print("错误：未能加载暗场帧")
        return None
    
    # 2. 执行标定
    black_levels = calibrator.calibrate(
        bayer_pattern=bayer_pattern,
        method='robust'  # 推荐使用robust方法
    )
    
    # 3. 时间稳定性分析
    calibrator.analyze_temporal_stability()
    
    # 4. 空间均匀性分析
    calibrator.analyze_spatial_uniformity(bayer_pattern)
    
    # 5. 生成报告
    calibrator.generate_report(output_folder)
    
    return black_levels


if __name__ == '__main__':
    # 使用示例
    black_levels = calibrate_black_level(
        dark_frame_folder='ISPpipline/raw_data/raw_dark_frames_long',
        output_folder='ISPpipline/black_level_report_long',
        width=1280,
        height=800,
        bayer_pattern='GBRG',
        dtype=np.uint16
    )
    
    if black_levels:
        print("\n✓ 标定完成！请查看 'black_level_calibration' 文件夹中的报告")