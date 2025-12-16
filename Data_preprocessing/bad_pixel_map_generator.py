# defect_map_generator.py
"""
坏点位置图生成工具
从全黑暗场帧中检测并标记Dead Pixel和Hot Pixel，生成报告图以及坏点地图给坏点校正使用。
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
    从暗场帧中检测坏点并生成位置图
    """
    
    def __init__(self):
        """初始化生成器"""
        self.dark_frames = []
        self.defect_map = None
        self.defect_types = None
    
    def load_dark_frames(self, dark_frame_folder: str,
                        width: int, height: int,
                        dtype: type = np.uint16,
                        pattern: str = '*.raw') -> int:
        """
        加载暗场帧
        
        Args:
            dark_frame_folder: 暗场帧文件夹
            width: 图像宽度
            height: 图像高度
            dtype: 数据类型
            pattern: 文件匹配模式
        
        Returns:
            加载的帧数
        """
        print("\n" + "="*70)
        print("加载暗场帧用于坏点检测")
        print("="*70)
        
        raw_files = sorted(glob.glob(os.path.join(dark_frame_folder, pattern)))
        
        if not raw_files:
            print(f"错误：未找到文件")
            return 0
        
        print(f"找到 {len(raw_files)} 个暗场帧")
        
        self.dark_frames = []
        for i, raw_file in enumerate(raw_files):
            try:
                with open(raw_file, 'rb') as f:
                    raw_data = np.fromfile(f, dtype=dtype)
                
                if raw_data.size == width * height:
                    frame = raw_data.reshape(height, width)
                    self.dark_frames.append(frame)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  已加载: {i+1}/{len(raw_files)}")
            except Exception as e:
                print(f"  错误: {e}")
                continue
        
        print(f"✓ 成功加载 {len(self.dark_frames)} 帧")
        return len(self.dark_frames)
    
    def detect_defects(self, method: str = 'statistical',
                      sensitivity: float = 1.0,
                      **kwargs) -> np.ndarray:
        """
        检测坏点并生成位置图
        
        Args:
            method: 检测方法
                - 'statistical': 统计方法（基于标准差）【推荐】
                - 'temporal': 时间域分析（需要多帧）
                - 'spatial': 空间域分析
                - 'hybrid': 混合方法（最全面）
            sensitivity: 灵敏度 (0.5-2.0)
                - < 1.0: 保守检测（少误报）
                - 1.0: 标准检测
                - > 1.0: 激进检测（找更多坏点）
        
        Returns:
            坏点位置图（bool数组，True=坏点）
        """
        if not self.dark_frames:
            raise ValueError("请先加载暗场帧")
        
        print("\n" + "="*70)
        print(f"坏点检测 - 方法: {method}, 灵敏度: {sensitivity}")
        print("="*70)
        
        if method == 'statistical':
            defect_map = self._detect_statistical(sensitivity, **kwargs)
        elif method == 'temporal':
            defect_map = self._detect_temporal(sensitivity, **kwargs)
        elif method == 'spatial':
            defect_map = self._detect_spatial(sensitivity, **kwargs)
        elif method == 'hybrid':
            defect_map = self._detect_hybrid(sensitivity, **kwargs)
        else:
            raise ValueError(f"未知方法: {method}")
        
        self.defect_map = defect_map
        
        # 分类坏点类型
        self.defect_types = self._classify_defects()
        
        # 统计
        num_defects = np.sum(defect_map)
        total_pixels = defect_map.size
        defect_ratio = num_defects / total_pixels * 100
        
        print(f"\n检测结果:")
        print(f"  总像素数: {total_pixels:,}")
        print(f"  坏点数量: {num_defects:,}")
        print(f"  坏点比例: {defect_ratio:.4f}%")
        print(f"  Dead Pixels (暗点): {self.defect_types['dead']:,}")
        print(f"  Hot Pixels (亮点):  {self.defect_types['hot']:,}")
        print(f"  Stuck Pixels (卡死): {self.defect_types['stuck']:,}")
        
        return defect_map
    
    def _detect_statistical(self, sensitivity: float,
                           sigma_threshold: float = None) -> np.ndarray:
        """
        统计方法：基于标准差检测（修复版）
        原理：坏点与周围像素的差异显著大于正常像素
        
        关键改进：
        1. Dead Pixel检测不依赖均值，而是检测绝对的低值和时间稳定性
        2. Hot Pixel检测使用百分位数而非均值+标准差
        3. 增加时间稳定性验证
        """
        print("\n使用统计方法检测...")
        
        # 1. 计算所有帧的平均和标准差
        dark_stack = np.stack(self.dark_frames, axis=0).astype(np.float32)
        avg_frame = np.mean(dark_stack, axis=0)
        std_frame = np.std(dark_stack, axis=0)
        median_frame = np.median(dark_stack, axis=0)
        
        print(f"  平均值范围: [{avg_frame.min():.2f}, {avg_frame.max():.2f}]")
        print(f"  中位数范围: [{median_frame.min():.2f}, {median_frame.max():.2f}]")
        print(f"  标准差范围: [{std_frame.min():.2f}, {std_frame.max():.2f}]")
        
        # 2. 计算局部统计量（用于检测局部异常）
        window_size = 5
        local_mean = uniform_filter(avg_frame, size=window_size)
        local_mean_sq = uniform_filter(avg_frame**2, size=window_size)
        local_var = local_mean_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # 3. 基于局部偏差的异常检测
        deviation = np.abs(avg_frame - local_mean)
        
        # 自适应阈值
        if sigma_threshold is None:
            sigma_threshold = 3.5 / sensitivity  # 提高基础阈值
        
        threshold = sigma_threshold * (local_std + 0.5)  # 减小偏移量
        
        # 初始异常点标记
        defect_map = deviation > threshold
        
        # ====================================================================
        # 4. Dead Pixel检测（关键修复！）
        # ====================================================================
        # 不能简单用均值判断，因为暗场图像本身就接近0
        # 改用以下标准：
        
        # 4.1 绝对低值检测（只检测真正的0或极低值）
        # 对于16bit数据，黑电平通常在10-50之间
        # Dead pixel应该 << 黑电平
        absolute_dead_threshold = 2.0  # 绝对阈值，接近0才算
        
        # 4.2 时间稳定性检测
        # Dead pixel的特征：值极低且时间方差接近0
        very_low_value = avg_frame < absolute_dead_threshold
        very_low_variance = std_frame < 0.5  # 时间方差极小
        
        # 同时满足才是Dead pixel
        dead_pixels = very_low_value & very_low_variance
        
        print(f"  Dead pixel检测:")
        print(f"    绝对阈值: {absolute_dead_threshold}")
        print(f"    检测到: {np.sum(dead_pixels)} 个")
        
        # ====================================================================
        # 5. Hot Pixel检测（改进版）
        # ====================================================================
        # 使用百分位数而非均值+标准差，更robust
        
        # 5.1 使用高百分位数作为正常上限
        p95 = np.percentile(avg_frame, 95)
        p99 = np.percentile(avg_frame, 99)
        p999 = np.percentile(avg_frame, 99.9)
        
        print(f"  百分位数分析:")
        print(f"    P95:  {p95:.2f}")
        print(f"    P99:  {p99:.2f}")
        print(f"    P99.9: {p999:.2f}")
        
        # 5.2 Hot pixel标准：
        # - 值 > P99（前1%最亮的）
        # - 且显著高于邻域
        # - 且时间方差较大（不稳定）或时间均值很高
        
        hot_value_threshold = p99 + (p999 - p99) * 0.5 / sensitivity
        
        # 相对邻域的偏差
        neighbor_deviation = avg_frame - local_mean
        hot_deviation_threshold = np.percentile(neighbor_deviation, 99) / sensitivity
        
        # Hot pixel候选
        hot_candidates = (avg_frame > hot_value_threshold) | \
                        (neighbor_deviation > hot_deviation_threshold)
        
        # 进一步验证：检查时间稳定性
        # Hot pixel通常有较大的时间方差或持续高值
        high_variance = std_frame > np.percentile(std_frame, 90)
        consistently_high = avg_frame > p95
        
        hot_pixels = hot_candidates & (high_variance | consistently_high)
        
        print(f"  Hot pixel检测:")
        print(f"    值阈值: {hot_value_threshold:.2f}")
        print(f"    检测到: {np.sum(hot_pixels)} 个")
        
        # ====================================================================
        # 6. 合并所有检测结果
        # ====================================================================
        defect_map = defect_map | dead_pixels | hot_pixels
        
        # ====================================================================
        # 7. 后处理：移除误报
        # ====================================================================
        # 7.1 孤立点滤除（单个孤立的检测可能是误报）
        from scipy.ndimage import binary_opening, generate_binary_structure
        
        # 使用形态学开运算去除孤立点
        struct = generate_binary_structure(2, 1)  # 4连通
        defect_map_filtered = binary_opening(defect_map, structure=struct)
        
        removed = np.sum(defect_map) - np.sum(defect_map_filtered)
        if removed > 0:
            print(f"  后处理: 移除 {removed} 个孤立误报点")
            defect_map = defect_map_filtered
        
        print(f"\n  最终检测: {np.sum(defect_map)} 个坏点")
        print(f"  坏点比例: {np.sum(defect_map)/defect_map.size*100:.4f}%")
        
        return defect_map
    
    def _detect_temporal(self, sensitivity: float,
                        variance_threshold: float = None) -> np.ndarray:
        """
        时间域分析：基于时间方差
        原理：坏点的时间方差异常（Dead Pixel方差为0，Hot Pixel方差大）
        """
        if len(self.dark_frames) < 3:
            print("警告: 时间域分析需要至少3帧，切换到统计方法")
            return self._detect_statistical(sensitivity)
        
        print("\n使用时间域方法检测...")
        
        dark_stack = np.stack(self.dark_frames, axis=0).astype(np.float32)
        
        # 1. 计算时间方差
        temporal_mean = np.mean(dark_stack, axis=0)
        temporal_var = np.var(dark_stack, axis=0)
        temporal_std = np.sqrt(temporal_var)
        
        print(f"  时间标准差范围: [{temporal_std.min():.2f}, {temporal_std.max():.2f}]")
        
        # 2. Dead Pixels: 时间方差接近0
        dead_threshold = temporal_std.mean() * 0.1 / sensitivity
        dead_pixels = temporal_std < dead_threshold
        
        # 3. Hot Pixels: 时间方差异常大或均值异常高
        hot_var_threshold = temporal_var.mean() + 3 * temporal_var.std() / sensitivity
        hot_mean_threshold = temporal_mean.mean() + 4 * temporal_mean.std() / sensitivity
        
        hot_pixels = (temporal_var > hot_var_threshold) | (temporal_mean > hot_mean_threshold)
        
        # 4. 组合
        defect_map = dead_pixels | hot_pixels
        
        print(f"  Dead检测阈值: {dead_threshold:.2f}")
        print(f"  Hot方差阈值: {hot_var_threshold:.2f}")
        
        return defect_map
    
    def _detect_spatial(self, sensitivity: float,
                       neighbor_threshold: float = None) -> np.ndarray:
        """
        空间域分析：基于邻域比较
        原理：坏点与相邻像素差异显著
        """
        print("\n使用空间域方法检测...")
        
        # 使用平均帧
        avg_frame = np.mean(self.dark_frames, axis=0).astype(np.float32)
        
        # 1. 计算与邻域中值的差异
        median_filtered = median_filter(avg_frame, size=5)
        spatial_diff = np.abs(avg_frame - median_filtered)
        
        # 2. 自适应阈值
        if neighbor_threshold is None:
            neighbor_threshold = 3.0 / sensitivity
        
        threshold = spatial_diff.mean() + neighbor_threshold * spatial_diff.std()
        
        defect_map = spatial_diff > threshold
        
        print(f"  空间差异阈值: {threshold:.2f}")
        
        return defect_map
    
    def _detect_hybrid(self, sensitivity: float) -> np.ndarray:
        """
        混合方法：结合多种检测方法
        最全面但可能有更多误报
        """
        print("\n使用混合方法检测...")
        
        # 1. 统计方法
        defect_stat = self._detect_statistical(sensitivity * 0.8)
        
        # 2. 时间域方法
        if len(self.dark_frames) >= 3:
            defect_temp = self._detect_temporal(sensitivity * 0.8)
        else:
            defect_temp = np.zeros_like(defect_stat, dtype=bool)
        
        # 3. 空间域方法
        defect_spat = self._detect_spatial(sensitivity * 0.8)
        
        # 4. 投票机制：至少2种方法检测到才算坏点
        vote_count = defect_stat.astype(int) + defect_temp.astype(int) + defect_spat.astype(int)
        
        defect_map = vote_count >= 2
        
        print(f"  统计方法检测: {np.sum(defect_stat):,}")
        print(f"  时间域检测:   {np.sum(defect_temp):,}")
        print(f"  空间域检测:   {np.sum(defect_spat):,}")
        print(f"  最终确认:     {np.sum(defect_map):,}")
        
        return defect_map
    
    def _classify_defects(self) -> dict:
        """
        分类坏点类型（修复版）
        
        Returns:
            {'dead': count, 'hot': count, 'stuck': count}
        """
        if self.defect_map is None:
            return {'dead': 0, 'hot': 0, 'stuck': 0}
        
        avg_frame = np.mean(self.dark_frames, axis=0)
        std_frame = np.std(self.dark_frames, axis=0)
        
        # 获取坏点位置
        defect_coords = np.where(self.defect_map)
        
        dead_count = 0
        hot_count = 0
        stuck_count = 0
        
        # 使用百分位数而非均值，避免被大量0值影响
        p50 = np.percentile(avg_frame, 50)  # 中位数
        p95 = np.percentile(avg_frame, 95)
        p99 = np.percentile(avg_frame, 99)
        
        print(f"\n  分类参考值:")
        print(f"    中位数(P50): {p50:.2f}")
        print(f"    P95: {p95:.2f}")
        print(f"    P99: {p99:.2f}")
        
        for y, x in zip(*defect_coords):
            pixel_value = avg_frame[y, x]
            pixel_std = std_frame[y, x]
            
            # Dead pixel: 值接近0且时间方差极小
            if pixel_value < 2.0 and pixel_std < 0.5:
                dead_count += 1
            # Hot pixel: 值 > P99 或显著高于中位数
            elif pixel_value > p99 or pixel_value > p50 + 3 * (p95 - p50):
                hot_count += 1
            # Stuck pixel: 其他异常像素
            else:
                stuck_count += 1
        
        return {
            'dead': dead_count,
            'hot': hot_count,
            'stuck': stuck_count
        }
    
    def save_defect_map(self, output_path: str = 'defect_map.npy'):
        """
        保存坏点位置图为numpy数组文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.defect_map is None:
            raise ValueError("请先执行检测")
        
        np.save(output_path, self.defect_map)
        print(f"\n✓ 坏点位置图已保存: {output_path}")
        print(f"  加载方式: defect_map = np.load('{output_path}')")
    
    def generate_report(self, output_folder: str = 'defect_map_analysis'):
        """
        生成完整的坏点分析报告（含可视化）
        
        Args:
            output_folder: 输出文件夹
        """
        if self.defect_map is None:
            raise ValueError("请先执行检测")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        print("\n" + "="*70)
        print("生成坏点分析报告")
        print("="*70)
        
        # 1. 保存坏点位置图
        map_path = os.path.join(output_folder, 'defect_map.npy')
        np.save(map_path, self.defect_map)
        print(f"✓ 坏点位置图: {map_path}")
        
        # 2. 生成文本报告
        report_path = os.path.join(output_folder, 'defect_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("坏点检测报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"图像尺寸: {self.defect_map.shape[1]} × {self.defect_map.shape[0]}\n")
            f.write(f"总像素数: {self.defect_map.size:,}\n")
            f.write(f"坏点数量: {np.sum(self.defect_map):,}\n")
            f.write(f"坏点比例: {np.sum(self.defect_map)/self.defect_map.size*100:.4f}%\n\n")
            
            f.write("坏点类型分布:\n")
            f.write(f"  Dead Pixels (死点):  {self.defect_types['dead']:,}\n")
            f.write(f"  Hot Pixels (亮点):   {self.defect_types['hot']:,}\n")
            f.write(f"  Stuck Pixels (其他异常像素): {self.defect_types['stuck']:,}\n\n")
            
            f.write("使用方法:\n\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("from defect_pixel_correction import DefectPixelCorrection\n\n")
            f.write("# 加载坏点位置图\n")
            f.write(f"defect_map = np.load('{map_path}')\n\n")
            f.write("# 在ISP流程中使用\n")
            f.write("processing_params = {\n")
            f.write("    'defectpixelcorrection': {\n")
            f.write("        'method': 'median',\n")
            f.write("        'defect_map': defect_map,\n")
            f.write("        'auto_detect': False\n")
            f.write("    }\n")
            f.write("}\n")
            f.write("```\n")
        
        print(f"✓ 文本报告: {report_path}")
        
        # 3. 生成可视化
        self._plot_analysis(output_folder)
        
        print(f"✓ 可视化图表已保存到: {output_folder}")
    
    def _plot_analysis(self, output_folder: str):
        """生成坏点分析图表"""
        fig = plt.figure(figsize=(16, 10))
        
        avg_frame = np.mean(self.dark_frames, axis=0)
        
        # 1. 原始平均暗场图
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(avg_frame, cmap='gray', vmin=0, vmax=np.percentile(avg_frame, 99))
        ax1.set_title('平均暗场图像', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='像素值')
        
        # 2. 坏点位置图
        ax2 = plt.subplot(2, 3, 2)
        defect_overlay = np.zeros((*self.defect_map.shape, 3), dtype=np.uint8)
        defect_overlay[self.defect_map] = [255, 0, 0]  # 红色标记坏点
        
        # 叠加显示
        base_img = (avg_frame / avg_frame.max() * 255).astype(np.uint8)
        base_rgb = np.stack([base_img]*3, axis=-1)
        overlay = cv2.addWeighted(base_rgb, 0.7, defect_overlay, 0.3, 0)
        
        ax2.imshow(overlay)
        ax2.set_title(f'坏点位置图 ({np.sum(self.defect_map):,}个坏点)', 
                     fontsize=14, fontweight='bold')
        
        # 3. 坏点类型分布
        ax3 = plt.subplot(2, 3, 3)
        types = ['Dead\nPixels', 'Hot\nPixels', 'Stuck\nPixels']
        counts = [self.defect_types['dead'], self.defect_types['hot'], self.defect_types['stuck']]
        colors = ['blue', 'red', 'orange']
        
        bars = ax3.bar(types, counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('数量', fontsize=12)
        ax3.set_title('坏点类型分布', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        # 4. 坏点空间分布热图
        ax4 = plt.subplot(2, 3, 4)
        # 将图像分成网格，统计每个区域的坏点数
        grid_size = 32
        h, w = self.defect_map.shape
        heat_map = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * h // grid_size
                y_end = (i + 1) * h // grid_size
                x_start = j * w // grid_size
                x_end = (j + 1) * w // grid_size
                
                heat_map[i, j] = np.sum(self.defect_map[y_start:y_end, x_start:x_end])
        
        im4 = ax4.imshow(heat_map, cmap='hot', interpolation='bilinear')
        ax4.set_title('坏点密度热图', fontsize=14, fontweight='bold')
        plt.colorbar(im4, ax=ax4, label='坏点数量')
        
        # 5. 坏点统计
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        stats_text = f"""
        统计信息：
        
        总像素数: {self.defect_map.size:,}
        坏点总数: {np.sum(self.defect_map):,}
        坏点比例: {np.sum(self.defect_map)/self.defect_map.size*100:.4f}%
        
        坏点类型：
        • Dead Pixels:  {self.defect_types['dead']:,}
        • Hot Pixels:   {self.defect_types['hot']:,}
        • Stuck Pixels: {self.defect_types['stuck']:,}
        
        质量评估：
        """
        
        defect_ratio = np.sum(self.defect_map) / self.defect_map.size * 100
        if defect_ratio < 0.01:
            quality = "优秀 ✓"
        elif defect_ratio < 0.05:
            quality = "良好"
        elif defect_ratio < 0.1:
            quality = "一般"
        else:
            quality = "较差 ⚠"
        
        stats_text += f"  传感器质量: {quality}"
        
        ax5.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center')
        
        # 6. 坏点值分布直方图
        ax6 = plt.subplot(2, 3, 6)
        defect_values = avg_frame[self.defect_map]
        
        ax6.hist(defect_values, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax6.axvline(avg_frame.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'整体均值: {avg_frame.mean():.1f}')
        ax6.set_xlabel('像素值', fontsize=12)
        ax6.set_ylabel('频数', fontsize=12)
        ax6.set_title('坏点值分布', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_folder, 'defect_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


# ===================================================================
# 使用示例
# ===================================================================

def generate_defect_map(dark_frame_folder: str,
                       output_folder:str,
                       width: int = 1280,
                       height: int = 800,
                       dtype: type = np.uint16,
                       method: str = 'hybrid',
                       sensitivity: float = 1.0):
    """
    从暗场帧生成坏点位置图
    
    Args:
        dark_frame_folder: 暗场帧文件夹
        width: 图像宽度
        height: 图像高度
        dtype: 数据类型
        method: 检测方法 ('statistical', 'temporal', 'spatial', 'hybrid')
        sensitivity: 灵敏度 (0.5-2.0)
    
    Returns:
        坏点位置图（bool数组）
    """
    # 创建生成器
    generator = DefectMapGenerator()
    
    # 1. 加载暗场帧
    num_frames = generator.load_dark_frames(
        dark_frame_folder,
        width,
        height,
        dtype
    )
    
    if num_frames == 0:
        print("错误：未能加载暗场帧")
        return None
    
    # 2. 检测坏点
    defect_map = generator.detect_defects(
        method=method,
        sensitivity=sensitivity
    )
    
    # 3. 生成报告
    generator.generate_report(output_folder)
    
    return defect_map


if __name__ == '__main__':
    # 使用示例
    defect_map = generate_defect_map(
        dark_frame_folder='ISPpipline/raw_data/raw_dark_frames_long',
        output_folder='ISPpipline/report/bad_points_report_long',
        width=1280,
        height=800,
        dtype=np.uint16,
        method='hybrid',      # 推荐使用混合方法
        sensitivity=1.0       # 标准灵敏度
    )
    
    if defect_map is not None:
        print("\n✓ 坏点检测完成！查看 'defect_map_analysis' 文件夹")