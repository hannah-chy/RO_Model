import json
import os
import pandas as pd
from datetime import datetime

class ExperimentLogger:
    """
    实验记录器类，用于管理实验结果的记录和比较。
    """
    def __init__(self, base_output_dir):
        """
        初始化实验记录器。
        
        参数:
        base_output_dir: 基础输出目录
        """
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        self.results_file = os.path.join(base_output_dir, 'experiment_results.json')
        self.comparison_file = os.path.join(base_output_dir, 'experiment_comparison.csv')

    def save_experiment(self, experiment_id, params, metrics, feature_info=None, group_by='none'):
        """
        保存单个实验的结果。
        
        参数:
        experiment_id: 实验ID
        params: 模型参数
        metrics: 评估指标
        feature_info: 特征工程信息
        group_by: 数据分组方式
        """
        experiment = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'group_by': group_by,
            'parameters': params,
            'metrics': metrics,
            'feature_engineering': feature_info or {
                'method': 'original',
                'features_used': [],
                'transformations': []
            }
        }
        
        # 读取现有实验记录
        try:
            with open(self.results_file, 'r') as f:
                experiments = json.load(f)
        except FileNotFoundError:
            experiments = []
        
        # 添加新实验结果
        experiments.append(experiment)
        
        # 保存更新后的实验记录
        with open(self.results_file, 'w') as f:
            json.dump(experiments, f, indent=4)
            
        return experiment

    def load_experiments(self):
        """
        加载所有实验记录。
        """
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def compare_experiments(self):
        """
        比较所有实验结果并生成汇总表。
        """
        experiments = self.load_experiments()
        if not experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in experiments:
            row = {
                'experiment_id': exp['experiment_id'],
                'timestamp': exp['timestamp'],
                'group_by': exp['group_by'],
                'feature_method': exp['feature_engineering']['method'],
                'train_r2': exp['metrics']['train']['r2'],
                'test_r2': exp['metrics']['test']['r2'],
                'train_mae': exp['metrics']['train']['mae'],
                'test_mae': exp['metrics']['test']['mae'],
                'train_mse': exp['metrics']['train']['mse'],
                'test_mse': exp['metrics']['test']['mse'],
                'features_used': ', '.join(exp['feature_engineering'].get('features_used', [])),
                'transformations': ', '.join(exp['feature_engineering'].get('transformations', []))
            }
            # 添加模型参数
            row.update({f"param_{k}": v for k, v in exp['parameters'].items()})
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 按测试集R2排序
        df_sorted = df.sort_values('test_r2', ascending=False)
        
        # 保存比较结果
        df_sorted.to_csv(self.comparison_file, index=False)
        
        return df_sorted

    def get_best_experiment(self, metric='test_r2'):
        """
        获取最佳实验结果。
        
        参数:
        metric: 用于选择最佳实验的指标
        """
        df = self.compare_experiments()
        if df.empty:
            return None
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()

    def summarize_experiments(self):
        """
        生成实验结果摘要。
        """
        df = self.compare_experiments()
        if df.empty:
            return "没有找到实验记录"
        
        summary = []
        summary.append("=== 实验结果摘要 ===")
        summary.append(f"总实验数: {len(df)}")
        
        # 按分组方式统计
        group_stats = df.groupby('group_by')['test_r2'].agg(['count', 'mean', 'max'])
        summary.append("\n按分组方式统计:")
        summary.append(group_stats.to_string())
        
        # 按特征工程方法统计
        feature_stats = df.groupby('feature_method')['test_r2'].agg(['count', 'mean', 'max'])
        summary.append("\n按特征工程方法统计:")
        summary.append(feature_stats.to_string())
        
        # 最佳实验
        best_exp = self.get_best_experiment()
        summary.append("\n最佳实验:")
        summary.append(f"ID: {best_exp['experiment_id']}")
        summary.append(f"分组方式: {best_exp['group_by']}")
        summary.append(f"特征工程: {best_exp['feature_method']}")
        summary.append(f"测试集 R2: {best_exp['test_r2']:.4f}")
        summary.append(f"测试集 MAE: {best_exp['test_mae']:.4f}")
        
        return "\n".join(summary)

def create_experiment_id(group_by, feature_method):
    """
    创建实验ID。
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{group_by}_{feature_method}_{timestamp}" 