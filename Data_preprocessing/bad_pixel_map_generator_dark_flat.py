# defect_map_generator.py
"""
坏点位置图生成工具 (ISP 级物理校准版)
- 从全黑暗场帧 (Dark Frames) 中检测亮点 (Hot Pixels)
- 从均匀亮场帧 (Flat Frames) 中检测暗点/死点 (Dead Pixels)
- 严格遵循 Bayer 阵列通道拆分，避免异色像素串扰
"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter
import cv2


class DefectMapGenerator:
    """
    坏点位置图生成器
    基于暗场和亮场双路检测，支持 Bayer 通道隔离
    """
    
    def __init__(self, bayer_pattern: str = 'RGGB'):
        """初始化生成器"""
        self.dark_frames = []
        self.flat_frames = []
        self.bayer_pattern = bayer_pattern.upper()
        
        self.defect_map = None
        self.hot_map = None
        self.dead_map = None
        
        # 验证 Bayer 模式
        valid_patterns = ['RGGB', 'BGGR', 'GBRG', 'GRBG']
        if self.bayer_pattern not in valid_patterns:
            raise ValueError(f"不支持的 Bayer 模式: {self.bayer_pattern}。支持: {valid_patterns}")

    # ====================================================================
    # 数据加载模块
    # ====================================================================

    def load_dark_frames(self, folder: str, width: int, height: int, dtype: type = np.uint16) -> int:
        """加载暗场帧 (用于检测亮点)"""
        print("\n" + "="*70)
        print("加载暗场帧 (Dark Frames) 用于 Hot Pixel 检测...")
        self.dark_frames = self._load_raw_folder(folder, width, height, dtype)
        return len(self.dark_frames)

    def load_flat_frames(self, folder: str, width: int, height: int, dtype: type = np.uint16) -> int:
        """加载亮场帧 (用于检测暗点)"""
        print("\n" + "="*70)
        print("加载亮场帧 (Flat Frames) 用于 Dead Pixel 检测...")
        self.flat_frames = self._load_raw_folder(folder, width, height, dtype)
        return len(self.flat_frames)

    def _load_raw_folder(self, folder: str, width: int, height: int, dtype: type, pattern: str = '*.raw') -> list:
        """通用的 RAW 文件加载逻辑"""
        raw_files = sorted(glob.glob(os.path.join(folder, pattern)))
        if not raw_files:
            print(f"警告：在 {folder} 中未找到文件")
            return []
        
        frames = []
        for i, raw_file in enumerate(raw_files):
            try:
                with open(raw_file, 'rb') as f:
                    raw_data = np.fromfile(f, dtype=dtype)
                if raw_data.size == width * height:
                    frame = raw_data.reshape(height, width)
                    frames.append(frame)
                    if (i + 1) % 10 == 0:
                        print(f"  已加载: {i+1}/{len(raw_files)}")
            except Exception as e:
                print(f"  错误: 加载 {raw_file} 失败 - {e}")
                
        print(f"✓ 成功加载 {len(frames)} 帧")
        return frames

    # ====================================================================
    # 核心算法：Bayer 通道隔离模块
    # ====================================================================

    def _split_bayer(self, image: np.ndarray) -> dict:
        """将整图按 Bayer 模式拆分为 4 个独立的颜色通道"""
        h, w = image.shape
        channels = {}
        
        if self.bayer_pattern == 'RGGB':
            channels['R']  = image[0:h:2, 0:w:2]
            channels['Gr'] = image[0:h:2, 1:w:2]
            channels['Gb'] = image[1:h:2, 0:w:2]
            channels['B']  = image[1:h:2, 1:w:2]
        elif self.bayer_pattern == 'BGGR':
            channels['B']  = image[0:h:2, 0:w:2]
            channels['Gb'] = image[0:h:2, 1:w:2]
            channels['Gr'] = image[1:h:2, 0:w:2]
            channels['R']  = image[1:h:2, 1:w:2]
        elif self.bayer_pattern == 'GBRG':
            channels['Gb'] = image[0:h:2, 0:w:2]
            channels['B']  = image[0:h:2, 1:w:2]
            channels['R']  = image[1:h:2, 0:w:2]
            channels['Gr'] = image[1:h:2, 1:w:2]
        elif self.bayer_pattern == 'GRBG':
            channels['Gr'] = image[0:h:2, 0:w:2]
            channels['R']  = image[0:h:2, 1:w:2]
            channels['B']  = image[1:h:2, 0:w:2]
            channels['Gb'] = image[1:h:2, 1:w:2]
            
        return channels

    def _merge_bayer(self, channels: dict, shape: tuple) -> np.ndarray:
        """将 4 个通道的检测结果 (bool 数组) 合并回原图分辨率"""
        h, w = shape
        merged = np.zeros((h, w), dtype=bool)
        
        if self.bayer_pattern == 'RGGB':
            merged[0:h:2, 0:w:2] = channels['R']
            merged[0:h:2, 1:w:2] = channels['Gr']
            merged[1:h:2, 0:w:2] = channels['Gb']
            merged[1:h:2, 1:w:2] = channels['B']
        elif self.bayer_pattern == 'BGGR':
            merged[0:h:2, 0:w:2] = channels['B']
            merged[0:h:2, 1:w:2] = channels['Gb']
            merged[1:h:2, 0:w:2] = channels['Gr']
            merged[1:h:2, 1:w:2] = channels['R']
        elif self.bayer_pattern == 'GBRG':
            merged[0:h:2, 0:w:2] = channels['Gb']
            merged[0:h:2, 1:w:2] = channels['B']
            merged[1:h:2, 0:w:2] = channels['R']
            merged[1:h:2, 1:w:2] = channels['Gr']
        elif self.bayer_pattern == 'GRBG':
            merged[0:h:2, 0:w:2] = channels['Gr']
            merged[0:h:2, 1:w:2] = channels['R']
            merged[1:h:2, 0:w:2] = channels['B']
            merged[1:h:2, 1:w:2] = channels['Gb']
            
        return merged

    # ====================================================================
    # 坏点检测模块
    # ====================================================================

    def detect_defects(self, sensitivity: float = 1.0) -> np.ndarray:
        """执行完整的双路坏点检测"""
        print("\n" + "="*70)
        print(f"开始执行坏点检测 - Bayer: {self.bayer_pattern}, 灵敏度: {sensitivity}")
        print("="*70)
        
        base_shape = None
        
        # 1. 亮点检测 (依赖 Dark Frames)
        if self.dark_frames:
            base_shape = self.dark_frames[0].shape
            self.hot_map = self._detect_hot_from_dark(sensitivity)
        else:
            print("  [跳过] 未加载暗场帧，无法检测 Hot Pixels。")
            
        # 2. 暗点检测 (依赖 Flat Frames)
        if self.flat_frames:
            if base_shape is None: base_shape = self.flat_frames[0].shape
            self.dead_map = self._detect_dead_from_flat(sensitivity)
        else:
            print("  [跳过] 未加载亮场帧，无法检测 Dead Pixels。")

        # 3. 合并 Map
        if base_shape is None:
            raise ValueError("请至少加载一组帧 (Dark 或 Flat) 才能进行检测")
            
        self.hot_map = self.hot_map if self.hot_map is not None else np.zeros(base_shape, dtype=bool)
        self.dead_map = self.dead_map if self.dead_map is not None else np.zeros(base_shape, dtype=bool)
        
        self.defect_map = self.hot_map | self.dead_map
        
        # 统计输出
        total_pixels = self.defect_map.size
        num_hot = np.sum(self.hot_map)
        num_dead = np.sum(self.dead_map)
        overlap = np.sum(self.hot_map & self.dead_map) # 极少情况：既是亮点又是暗点（通常是闪烁噪点）
        
        print(f"\n检测完成汇总:")
        print(f"  总像素数: {total_pixels:,}")
        print(f"  Hot Pixels (亮点):  {num_hot:,}")
        print(f"  Dead Pixels (暗点): {num_dead:,}")
        print(f"  总独立坏点数: {np.sum(self.defect_map):,} (占比 {np.sum(self.defect_map)/total_pixels*100:.4f}%)")
        
        return self.defect_map

    def _detect_hot_from_dark(self, sensitivity: float) -> np.ndarray:
        """从暗场帧检测漏电流亮点 (严格在单颜色通道内进行局部统计)"""
        print("\n  > 分析暗场帧寻找 Hot Pixels...")
        dark_stack = np.stack(self.dark_frames, axis=0).astype(np.float32)
        avg_dark = np.mean(dark_stack, axis=0)
        
        # 拆分通道
        channels_avg = self._split_bayer(avg_dark)
        channels_hot_map = {}
        
        for ch_name, ch_data in channels_avg.items():
            # 计算局部中值和局部标准差 (在同色像素网格内，窗口 size=3 相当于原图 6x6 范围)
            local_median = median_filter(ch_data, size=3)
            local_mean = uniform_filter(ch_data, size=3)
            local_var = uniform_filter(ch_data**2, size=3) - local_mean**2
            local_std = np.sqrt(np.maximum(local_var, 0))
            
            # 判据 1：绝对阈值 (超过该通道 99.9% 亮度的绝对值)
            p999 = np.percentile(ch_data, 99.9)
            abs_threshold = p999 * (1.2 / sensitivity)
            
            # 判据 2：相对局部异常 (比周围同色像素显著亮出 5 个标准差)
            rel_threshold = local_median + (5.0 / sensitivity) * local_std
            
            # 满足其一且显著高于黑电平的即为 Hot Pixel
            channels_hot_map[ch_name] = (ch_data > abs_threshold) | ((ch_data > rel_threshold) & (ch_data > local_median * 1.5))
            
        return self._merge_bayer(channels_hot_map, avg_dark.shape)

    def _detect_dead_from_flat(self, sensitivity: float) -> np.ndarray:
        """从亮场帧检测死点/暗点 (严格在单颜色通道内进行局部统计)"""
        print("\n  > 分析亮场帧寻找 Dead Pixels...")
        flat_stack = np.stack(self.flat_frames, axis=0).astype(np.float32)
        avg_flat = np.mean(flat_stack, axis=0)
        
        # 拆分通道
        channels_avg = self._split_bayer(avg_flat)
        channels_dead_map = {}
        
        for ch_name, ch_data in channels_avg.items():
            global_median = np.median(ch_data)
            local_median = median_filter(ch_data, size=3)
            
            # 判据 1：绝对死点 (无论给多少光，数值都极低，接近黑电平)
            # 假设亮场正常曝光在全量程的 50% 以上，那么低于全局中位数的 20% 视为异常低
            abs_dead_threshold = global_median * 0.2 * sensitivity
            
            # 判据 2：相对暗点 (比周围同色像素暗了一半以上)
            rel_dead_threshold = local_median * 0.5 * sensitivity
            
            channels_dead_map[ch_name] = (ch_data < abs_dead_threshold) | (ch_data < rel_dead_threshold)
            
        return self._merge_bayer(channels_dead_map, avg_flat.shape)

    # ====================================================================
    # 报告与可视化模块
    # ====================================================================

    def save_defect_map(self, output_path: str = 'defect_map.npy'):
        if self.defect_map is None: raise ValueError("请先执行检测")
        np.save(output_path, self.defect_map)
        print(f"\n✓ 坏点位置图已保存: {output_path}")

    def generate_report(self, output_folder: str = 'defect_map_analysis'):
        """生成报告 (适配新的双路统计)"""
        if self.defect_map is None: raise ValueError("请先执行检测")
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        # 保存 Map
        map_path = os.path.join(output_folder, 'defect_map.npy')
        np.save(map_path, self.defect_map)
        
        # 文本报告
        report_path = os.path.join(output_folder, 'defect_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n坏点检测报告 (双路 ISP 标准版)\n" + "="*70 + "\n\n")
            f.write(f"Bayer 模式: {self.bayer_pattern}\n")
            f.write(f"图像尺寸: {self.defect_map.shape[1]} × {self.defect_map.shape[0]}\n")
            f.write(f"总像素数: {self.defect_map.size:,}\n")
            f.write(f"总坏点数: {np.sum(self.defect_map):,} ({np.sum(self.defect_map)/self.defect_map.size*100:.4f}%)\n\n")
            f.write("坏点分类统计:\n")
            f.write(f"  Hot Pixels (来自暗场): {np.sum(self.hot_map):,}\n")
            f.write(f"  Dead Pixels (来自亮场): {np.sum(self.dead_map):,}\n")
            
        print(f"✓ 分析报告已生成: {output_folder}")


# ===================================================================
# 使用示例
# ===================================================================

def generate_defect_map_pipeline():
    """执行双路物理坏点校准流水线"""
    
    # 初始化，指定传感器的 Bayer 排列方式
    # 如果不确定，通常是 RGGB 或 BGGR
    generator = DefectMapGenerator(bayer_pattern='RGGB')
    
    # 1. 加载暗场数据 (盖紧镜头盖拍摄)
    generator.load_dark_frames(
        folder='ISPpipline/raw_data/raw_dark_frames_long',
        width=1280, height=800, dtype=np.uint16
    )
    
    # 2. 加载亮场数据 (对着均匀光源或白墙拍摄)
    # 建议准备这个文件夹和数据，即使只有一两帧也能极大提升死点检测准确度
    generator.load_flat_frames(
        folder='ISPpipline/raw_data/raw_flat_frames', 
        width=1280, height=800, dtype=np.uint16
    )
    
    # 3. 核心检测 (内部已处理通道拆分和双路判定)
    defect_map = generator.detect_defects(sensitivity=1.0)
    
    if defect_map is not None:
        generator.generate_report(output_folder='ISPpipline/report/bad_points_report_long')
        print("\n✓ 物理校准流程完成！")

if __name__ == '__main__':
    generate_defect_map_pipeline()