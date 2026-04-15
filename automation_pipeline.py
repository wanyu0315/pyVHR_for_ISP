#!/usr/bin/env python3
"""
ISP-pyVHR 自动化集成脚本
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

    def __init__(
        self,
        baseline_params: dict,
        sweep_config: dict = None,
        single_run_variant_name: str = "baseline"
    ):
        self.baseline_params = baseline_params
        self.sweep_config = sweep_config
        self.single_run_variant_name = single_run_variant_name or "baseline"

    @staticmethod
    def get_param_label(param_path: str) -> str:
        """返回用于目录和分组命名的参数标签。"""
        if not param_path:
            return "param"

        normalized = param_path.replace(']', '').replace('[', '.')
        parts = [p for p in normalized.split('.') if p and not p.isdigit()]
        return parts[-1] if parts else param_path

    @staticmethod
    def _parse_param_path(param_path: str) -> List[object]:
        """
        解析参数路径，支持以下形式：
        - gamma
        - steps.0.alpha
        - steps[0].alpha
        - steps[temporal].alpha
        - steps[algorithm=temporal].alpha
        """
        tokens: List[object] = []

        for segment in param_path.split('.'):
            current = segment
            while current:
                if '[' not in current:
                    tokens.append(int(current) if current.isdigit() else current)
                    break

                prefix, suffix = current.split('[', 1)
                if prefix:
                    tokens.append(prefix)

                selector, remainder = suffix.split(']', 1)
                if selector.isdigit():
                    tokens.append(int(selector))
                elif '=' in selector:
                    field, value = selector.split('=', 1)
                    tokens.append(('field_match', field, value))
                else:
                    tokens.append(('field_match', 'algorithm', selector))

                current = remainder
                if current.startswith('.'):
                    current = current[1:]

        return tokens

    @staticmethod
    def _resolve_list_selector(items: list, selector: object) -> object:
        """在列表中解析索引或按字段匹配的选择器。"""
        if isinstance(selector, int):
            return items[selector]

        if isinstance(selector, tuple) and selector[0] == 'field_match':
            _, field, expected = selector
            for item in items:
                if isinstance(item, dict) and str(item.get(field)) == expected:
                    return item
            raise KeyError(f"列表中不存在 {field}={expected} 的步骤")

        raise TypeError(f"不支持的列表选择器: {selector}")

    @classmethod
    def _set_nested_param(cls, module_params: dict, param_path: str, value):
        """设置模块内部的嵌套参数值。"""
        tokens = cls._parse_param_path(param_path)
        if not tokens:
            raise ValueError("参数路径不能为空")

        current = module_params
        for token in tokens[:-1]:
            if isinstance(current, dict):
                current = current[token]
            elif isinstance(current, list):
                current = cls._resolve_list_selector(current, token)
            else:
                raise TypeError(f"无法继续解析参数路径 '{param_path}'")

        last_token = tokens[-1]
        if isinstance(current, dict):
            current[last_token] = value
        elif isinstance(current, list):
            if isinstance(last_token, int):
                current[last_token] = value
            else:
                raise TypeError(f"列表末端只支持整数索引赋值: {param_path}")
        else:
            raise TypeError(f"无法在目标对象上设置参数 '{param_path}'")

    def generate_variants(self) -> List[Tuple[str, dict]]:
        """
        生成参数变体列表

        当 sweep_config 为 None 时，直接将 baseline_params 作为唯一变体运行一次。
        变体名称可由配置指定，默认使用 "baseline"。
        """
        # 无参数扫描：单次运行 baseline
        if self.sweep_config is None:
            return [(self.single_run_variant_name, copy.deepcopy(self.baseline_params))]

        target_module = self.sweep_config['target_module']
        target_param = self.sweep_config['target_param']
        values = self.sweep_config['values']

        variants = []

        for value in values:
            params = copy.deepcopy(self.baseline_params)

            if target_module in params:
                try:
                    self._set_nested_param(params[target_module], target_param, value)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    print(
                        f"警告: 参数路径 '{target_module}.{target_param}' 设置失败: {exc}"
                    )
                    continue
            else:
                print(f"警告: 模块 '{target_module}' 不存在于 baseline_params 中")
                continue

            variant_name = f"{value}"
            variants.append((variant_name, params))

        return variants


class VideoGroupsBuilder:
    """video_groups 自动构建器"""

    def __init__(self, gt_data_dir: str, subjects: List[str]):
        self.gt_data_dir = gt_data_dir
        self.subjects = subjects

    def build_from_isp_output(
        self,
        isp_output_dir: str,
        group_name_prefix: str,
        output_bit_depth: int = 8
    ) -> dict:
        """从 ISP 输出目录构建 video_group"""
        videos = []

        for subject in self.subjects:
            video_filename = f"{subject}_output_{output_bit_depth}bit.mkv"
            video_path = os.path.join(isp_output_dir, video_filename)

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

    BASELINE_VIDEO_GROUP_DIR = "Data_for_pyVHR/isp_output_Video/baseenv_baselineISP"

    def __init__(self, config_path: str):
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
        sweep_config = self.config['isp'].get('parameter_sweep', None)
        single_run_dirname = self.config['output'].get('single_run_dirname', 'baseline')
        generator = ISPParamVariantGenerator(
            baseline_params=self.config['isp']['baseline_params'],
            sweep_config=sweep_config,
            single_run_variant_name=single_run_dirname
        )
        variants = generator.generate_variants()

        if sweep_config is None:
            print(f"  ✓ 未配置参数扫描，以单次参数运行，输出目录名: {single_run_dirname}")
        else:
            print(f"  ✓ 生成 {len(variants)} 个参数变体")

        # 准备路径
        # 有扫描时：ISP 视频/帧路径含 experiment_name 和参数维度
        # 例如 .../baseenv/rawdenoise_study/alpha/0.4/
        # 无扫描时：路径使用 experiment_name 作为终端目录（如 .../rawdenoise_study/baseline_run/）
        if sweep_config is not None:
            target_param = sweep_config['target_param']
            path_param_dim = ISPParamVariantGenerator.get_param_label(target_param)
        else:
            target_param = None
            path_param_dim = self.experiment_name

        if sweep_config is not None:
            isp_video_base = os.path.join(
                self.config['output']['isp_video_base'],
                self.experiment_type,
                'baseenv',
                self.experiment_name,
                path_param_dim
            )

            isp_frame_base = os.path.join(
                self.config['output']['isp_frame_base'],
                self.experiment_type,
                'baseenv',
                self.experiment_name,
                path_param_dim
            )
        else:
            isp_video_base = os.path.join(
                self.config['output']['isp_video_base'],
                self.experiment_type,
                'baseenv',
                path_param_dim
            )

            isp_frame_base = os.path.join(
                self.config['output']['isp_frame_base'],
                self.experiment_type,
                'baseenv',
                path_param_dim
            )

        if sweep_config is not None:
            analysis_output_base = os.path.join(
                self.config['output']['analysis_results_base'],
                self.experiment_type,
                self.experiment_name,
                path_param_dim,
                f"{self.config['pyvhr']['analysis_params']['rppg_method']}"
            )
        else:
            analysis_output_base = os.path.join(
                self.config['output']['analysis_results_base'],
                self.experiment_type,
                path_param_dim,
                f"{self.config['pyvhr']['analysis_params']['rppg_method']}"
            )

        all_video_groups = []
        builder = VideoGroupsBuilder(
            gt_data_dir=self.config['pyvhr']['gt_data_dir'],
            subjects=self.config['pyvhr']['subjects']
        )
        probe_config = copy.deepcopy(self.config['isp'].get('probe_system', None))
        if probe_config is not None:
            probe_output_dirname = self.config['output'].get('probes_output_dirname')
            if probe_output_dirname is not None:
                probe_config['probes_output_dirname'] = probe_output_dirname
            probe_output_subdir = self.config['output'].get('probes_output_subdir')
            if probe_output_subdir is not None:
                probe_config['probes_output_subdir'] = probe_output_subdir

        # 无论是否执行参数扫描，最终 pyVHR 分析都始终包含固定的 baseline 视频组。
        print(f"\n  → 加载固定 baseline 视频组...")
        baseline_video_group = builder.build_from_isp_output(
            isp_output_dir=self.BASELINE_VIDEO_GROUP_DIR,
            group_name_prefix="baseenv_baselineISP",
            output_bit_depth=self.config['isp']['output_bit_depth']
        )
        if baseline_video_group['videos']:
            all_video_groups.append(baseline_video_group)
            print(f"  ✓ baseline 视频组已加入，包含 {len(baseline_video_group['videos'])} 个视频")
        else:
            print(f"  ⚠️ baseline 视频组为空: {self.BASELINE_VIDEO_GROUP_DIR}")

        # Step 2-3: 对每个参数变体运行 ISP 和构建 video_groups
        for idx, (variant_name, processing_params) in enumerate(variants):
            print(f"\n{'='*80}")
            print(f"[步骤 2/4] 处理参数变体 [{idx+1}/{len(variants)}]: {variant_name}")
            print(f"{'='*80}")

            variant_video_dir = os.path.join(isp_video_base, variant_name)
            variant_frame_dir = os.path.join(isp_frame_base, variant_name)

            # 运行 ISP 处理（传递探针配置）
            print(f"\n  → 运行 ISP 处理...")
            generated_videos = run_isp_pipeline(
                input_dir=self.config['isp']['input_raw_dir'],
                output_frame_dir=variant_frame_dir,
                output_video_dir=variant_video_dir,
                processing_params=processing_params,
                probe_config=probe_config,
                output_bit_depth=self.config['isp']['output_bit_depth'],
                image_width=self.config['isp']['sensor']['width'],
                image_height=self.config['isp']['sensor']['height'],
                bayer_pattern=self.config['isp']['sensor']['bayer_pattern']
            )

            print(f"  ✓ ISP 处理完成，生成 {len(generated_videos)} 个视频")

            # 构建 video_group
            print(f"\n  → 构建 video_group...")
            group_name_prefix = (
                f"baseenv_{variant_name}{ISPParamVariantGenerator.get_param_label(target_param)}ISP"
                if target_param is not None
                else f"baseenv_{self.experiment_name}ISP"
            )
            video_group = builder.build_from_isp_output(
                isp_output_dir=variant_video_dir,
                group_name_prefix=group_name_prefix,
                output_bit_depth=self.config['isp']['output_bit_depth']
            )

            if os.path.normpath(variant_video_dir) == os.path.normpath(self.BASELINE_VIDEO_GROUP_DIR):
                print("  ↪ 当前实验组与固定 baseline 目录相同，跳过重复加入。")
            else:
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
