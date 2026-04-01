#!/usr/bin/env python3
"""
ISP-pyVHR 自动化集成脚本

功能：
1. 根据配置文件生成 ISP 参数变体
2. 批量运行 ISP 处理
3. 自动构建 video_groups
4. 批量运行 pyVHR 分析
5. 生成对比结果

使用方法：
    python automation_pipeline.py --config automation_config.yaml
"""

import os
import sys
import yaml
import copy
import glob
from typing import List, Tuple, Dict
from datetime import datetime

# 导入 ISP 和 pyVHR 的封装函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ISPpipline'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyVHR_run_on_video'))

from ISPpipline.ISP_main import run_isp_pipeline
from pyVHR_run_on_video.analyze_with_pyvhr import run_pyvhr_analysis


class ISPParamVariantGenerator:
    """ISP 参数变体生成器"""

    def __init__(self, baseline_params: dict, sweep_config: dict):
        """
        Parameters:
        -----------
        baseline_params : dict
            基线 ISP 参数
        sweep_config : dict
            参数扫描配置，包含 target_module, target_param, values
        """
        self.baseline_params = baseline_params
        self.sweep_config = sweep_config

    def generate_variants(self) -> List[Tuple[str, dict]]:
        """
        生成参数变体列表

        Returns:
        --------
        List[Tuple[str, dict]] : [(variant_name, processing_params), ...]
        """
        target_module = self.sweep_config['target_module']
        target_param = self.sweep_config['target_param']
        values = self.sweep_config['values']

        variants = []

        for value in values:
            # 深拷贝基线参数
            params = copy.deepcopy(self.baseline_params)

            # 修改目标参数
            if target_module in params:
                params[target_module][target_param] = value
            else:
                print(f"警告: 模块 '{target_module}' 不存在于 baseline_params 中")
                continue

            # 生成变体名称
            variant_name = f"{value}"

            variants.append((variant_name, params))

        return variants


class VideoGroupsBuilder:
    """video_groups 自动构建器"""

    def __init__(self, gt_data_dir: str, subjects: List[str]):
        """
        Parameters:
        -----------
        gt_data_dir : str
            GT 数据根目录
        subjects : List[str]
            受试者列表
        """
        self.gt_data_dir = gt_data_dir
        self.subjects = subjects

    def build_from_isp_output(
        self,
        isp_output_dir: str,
        group_name_prefix: str,
        output_bit_depth: int = 8
    ) -> dict:
        """
        从 ISP 输出目录构建 video_group

        Parameters:
        -----------
        isp_output_dir : str
            ISP 输出视频目录
        group_name_prefix : str
            组名前缀（如 'baseenv_0.8gammaISP'）
        output_bit_depth : int
            输出位深

        Returns:
        --------
        dict : video_group 配置
        """
        videos = []

        for subject in self.subjects:
            video_filename = f"{subject}_output_{output_bit_depth}bit.mkv"
            video_path = os.path.join(isp_output_dir, video_filename)

            # 检查视频是否存在
            if not os.path.exists(video_path):
                print(f"  警告: 视频文件不存在: {video_path}")
                continue

            gt_path = os.path.join(self.gt_data_dir, f"gt_{subject}", "bpms_times_GT")

            videos.append({
                'video_path': video_path,
                'gt_path': gt_path,
                'name': f"{group_name_prefix}_{subject}"
            })

        group_name = f"{group_name_prefix}_VG"

        return {
            'group_name': group_name,
            'videos': videos
        }


class AutomationPipeline:
    """自动化流程主控制器"""

    def __init__(self, config_path: str):
        """
        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.experiment_name = self.config['experiment_name']
        self.experiment_type = self.config['experiment_type']

    def run(self):
        """运行完整的自动化流程"""

        print(f"\n{'='*80}")
        print(f"ISP-pyVHR 自动化集成系统")
        print(f"{'='*80}")
        print(f"实验名称: {self.experiment_name}")
        print(f"实验类型: {self.experiment_type}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Step 1: 生成参数变体
        print(f"\n[步骤 1/4] 生成 ISP 参数变体...")
        generator = ISPParamVariantGenerator(
            baseline_params=self.config['isp']['baseline_params'],
            sweep_config=self.config['isp']['parameter_sweep']
        )
        variants = generator.generate_variants()
        print(f"  ✓ 生成 {len(variants)} 个参数变体")

        # 准备路径
        target_module = self.config['isp']['parameter_sweep']['target_module']
        target_param = self.config['isp']['parameter_sweep']['target_param']

        # 构建输出路径
        isp_video_base = os.path.join(
            self.config['output']['isp_video_base'],
            self.experiment_type,
            'baseenv',
            target_param
        )

        isp_frame_base = os.path.join(
            self.config['output']['isp_frame_base'],
            self.experiment_type,
            'baseenv',
            target_param
        )

        analysis_output_base = os.path.join(
            self.config['output']['analysis_results_base'],
            self.experiment_type,
            target_param,
            f"{self.config['pyvhr']['analysis_params']['rppg_method']}_v1"
        )

        all_video_groups = []

        # Step 2-3: 对每个参数变体运行 ISP 和构建 video_groups
        for idx, (variant_name, processing_params) in enumerate(variants):
            print(f"\n{'='*80}")
            print(f"[步骤 2/4] 处理参数变体 [{idx+1}/{len(variants)}]: {variant_name}")
            print(f"{'='*80}")

            # ISP 输出路径
            variant_video_dir = os.path.join(isp_video_base, variant_name)
            variant_frame_dir = os.path.join(isp_frame_base, variant_name)

            # 运行 ISP 处理
            print(f"\n  → 运行 ISP 处理...")
            generated_videos = run_isp_pipeline(
                input_dir=self.config['isp']['input_raw_dir'],
                output_frame_dir=variant_frame_dir,
                output_video_dir=variant_video_dir,
                processing_params=processing_params,
                output_bit_depth=self.config['isp']['output_bit_depth'],
                image_width=self.config['isp']['sensor']['width'],
                image_height=self.config['isp']['sensor']['height'],
                bayer_pattern=self.config['isp']['sensor']['bayer_pattern']
            )

            print(f"  ✓ ISP 处理完成，生成 {len(generated_videos)} 个视频")

            # 构建 video_group
            print(f"\n  → 构建 video_group...")
            builder = VideoGroupsBuilder(
                gt_data_dir=self.config['pyvhr']['gt_data_dir'],
                subjects=self.config['pyvhr']['subjects']
            )

            group_name_prefix = f"baseenv_{variant_name}{target_param}ISP"
            video_group = builder.build_from_isp_output(
                isp_output_dir=variant_video_dir,
                group_name_prefix=group_name_prefix,
                output_bit_depth=self.config['isp']['output_bit_depth']
            )

            all_video_groups.append(video_group)
            print(f"  ✓ video_group 构建完成，包含 {len(video_group['videos'])} 个视频")

        # Step 4: 运行 pyVHR 批量分析
        print(f"\n{'='*80}")
        print(f"[步骤 3/4] 运行 pyVHR 批量分析")
        print(f"{'='*80}")

        result = run_pyvhr_analysis(
            video_groups=all_video_groups,
            analysis_params=self.config['pyvhr']['analysis_params'],
            output_dir=analysis_output_base
        )

        print(f"\n{'='*80}")
        print(f"[步骤 4/4] 自动化流程完成")
        print(f"{'='*80}")
        print(f"  ✓ 处理参数变体数: {len(variants)}")
        print(f"  ✓ 处理视频组数: {result['total_groups']}")
        print(f"  ✓ 处理视频总数: {result['total_videos']}")
        print(f"  ✓ 结果输出目录: {result['output_dir']}")
        print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='ISP-pyVHR 自动化集成系统')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (YAML 格式)')

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    pipeline = AutomationPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
