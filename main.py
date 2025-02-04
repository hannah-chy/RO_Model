from data_processor import load_and_split_data, apply_feature_engineering
from model_trainer import train_and_evaluate_model_with_optimization, evaluate_on_test_data
from visualizer import Visualizer
from experiment_logger import ExperimentLogger, create_experiment_id
from mechanism_contributions import MechanismAnalyzer, run_analysis
import os
from utils import validate_mechanism_config
import json
import matplotlib

class ExperimentManager:
    """
    实验管理器：整合数据处理、模型训练、结果记录和可视化
    """
    def __init__(self, config):
        """
        初始化实验管理器
        """
        self.config = config
        self.logger = ExperimentLogger(config['base_output_dir'])
        
        # 添加：验证机制配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'mechanism_config.json')
        self.mechanism_config_valid = validate_mechanism_config(config_path)
        if not self.mechanism_config_valid:
            print("警告：机制配置文件验证失败，将使用默认配置")
            
        # 添加：验证特征列表
        self._validate_features()
    
    def _validate_features(self):
        """验证特征列表与机制配置的一致性"""
        try:
            with open(os.path.join(os.path.dirname(__file__), 'mechanism_config.json'), 'r') as f:
                mechanism_config = json.load(f)
            
            # 检查特征重复
            from utils import check_feature_duplicates
            duplicates = check_feature_duplicates(mechanism_config)
            if duplicates:
                print("\n警告：检测到特征重复问题")
                print("建议修改mechanism_config.json，确保每个特征只出现在一个机制组中")
                print("当前重复特征及其所在机制组：")
                for feature, groups in duplicates.items():
                    print(f"  - {feature}: {', '.join(groups)}")
            
            # 获取配置文件中的所有特征
            config_features = set()
            for group_info in mechanism_config['mechanism_groups'].values():
                config_features.update(group_info['features'])
            
            # 检查是否所有配置的特征都在输入特征列表中
            missing_features = config_features - set(self.config['features'])
            if missing_features:
                print(f"\n警告：以下特征在输入特征列表中缺失: {missing_features}")
                
            # 检查是否有未使用的输入特征
            unused_features = set(self.config['features']) - config_features
            if unused_features:
                print(f"\n警告：以下输入特征未在机制配置中使用: {unused_features}")
                
        except Exception as e:
            print(f"特征验证失败: {str(e)}")
    
    def run_single_experiment(self, group_by, feature_method):
        """
        运行单个实验
        """
        # 1. 创建实验ID和输出目录
        experiment_id = create_experiment_id(group_by, feature_method)
        experiment_dir = os.path.join(self.config['base_output_dir'], experiment_id)
        print(f"\n=== 运行实验 {experiment_id} ===")
        
        # 2. 数据处理
        train_val_data, test_data = load_and_split_data(
            self.config['file_path'], 
            group_by=group_by
        )
        
        # 3. 特征工程
        feature_names = self.config['features']  # 统一特征名参数
        method = feature_method  # 统一方法参数

        # 对训练集进行特征工程,并保存特征名列表
        X_train_processed, feature_info_train = apply_feature_engineering(
            train_val_data[feature_names], 
            method=method
        )
        train_feature_names = feature_info_train['feature_names']
        
        # 使用相同的特征名列表对测试集进行特征工程
        X_test_processed, feature_info_test = apply_feature_engineering(
            test_data[feature_names], 
            method=method,
            feature_names=train_feature_names  # 使用训练集的特征名列表
        )
        
        # 4. 模型训练和评估
        model, avg_mse, avg_mae, avg_r2, y_train_true, y_train_pred = \
            train_and_evaluate_model_with_optimization(
                X_train_processed,
                train_val_data[self.config['target']],
                feature_names=train_feature_names  # 使用训练集的特征名列表
            )
        
        # 5. 测试集评估
        test_mse, test_mae, test_r2, y_test_true, y_test_pred = \
            evaluate_on_test_data(
                model, 
                X_test_processed,
                test_data[self.config['target']]
            )
        
        # 6. 机制分析（新增）
        mechanism_dir = os.path.join(experiment_dir, 'mechanism_analysis')
        analysis_results, mechanism_report = run_analysis(
            model=model,
            data=X_test_processed,  # 使用处理后的测试集数据
            output_dir=mechanism_dir
        )
        
        # 7. 记录实验结果（更新部分）
        metrics = {
            'train': {
                'mse': avg_mse,
                'mae': avg_mae,
                'r2': avg_r2
            },
            'test': {
                'mse': test_mse,
                'mae': test_mae,
                'r2': test_r2
            },
            'mechanism_analysis': mechanism_report
        }
        
        self.logger.save_experiment(
            experiment_id,
            model.get_params(),
            metrics,
            feature_info_train,
            group_by
        )
        
        # 8. 生成可视化结果
        viz_metrics = {
            'y_train_true': y_train_true,
            'y_train_pred': y_train_pred,
            'y_test_true': y_test_true,
            'y_test_pred': y_test_pred,
            'train_mae': avg_mae,
            'test_mae': test_mae,
            'train_r2': avg_r2,
            'test_r2': test_r2
        }
        
        # 创建可视化器
        visualizer = Visualizer(experiment_dir)
        
        # 绘制模型结果
        visualizer.plot_model_results(
            model, 
            X_test_processed,
            train_feature_names,  # 使用训练集的特征名列表
            viz_metrics
        )
        
        # 绘制机制分析结果
        visualizer.plot_mechanism_analysis(analysis_results)
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_info': feature_info_train,
            'mechanism_results': analysis_results
        }
    
    def run_all_experiments(self):
        """
        运行所有实验组合
        """
        results = {}
        for group_by in self.config['group_methods']:
            for feature_method in self.config['feature_methods']:
                results[f"{group_by}_{feature_method}"] = \
                    self.run_single_experiment(group_by, feature_method)
        
        # 输出实验摘要
        print("\n=== 实验结果摘要 ===")
        summary = self.logger.summarize_experiments()
        print(summary)
        
        # 添加机制分析摘要（新增）
        print("\n=== 机制分析摘要 ===")
        best_exp = self.logger.get_best_experiment()
        best_results = results[f"{best_exp['group_by']}_{best_exp['feature_method']}"]
        
        # 打印机制贡献度
        print("\n最佳模型的机制贡献度:")
        mechanism_scores = best_results['mechanism_results']['mechanism_contributions']
        for mechanism, details in mechanism_scores.items():
            print(f"{mechanism}:")
            print(f"  总得分: {details['total_score']:.4f}")
            print(f"  SHAP贡献: {details['shap_contribution']:.4f}")
            print(f"  特征重要性: {details['feature_importance']:.4f}")
        
        # 打印机制交互
        print("\n机制交互强度:")
        interactions = best_results['mechanism_results']['mechanism_interactions']
        for pair, strength in interactions.items():
            print(f"{pair}: {strength:.4f}")
        
        return results

def main():
    try:
        # 设置matplotlib后端
        matplotlib.use('Agg')
        
        # 验证配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'mechanism_config.json')
        if not validate_mechanism_config(config_path):
            print("配置文件验证失败，将使用默认配置")
        
        # 1. 配置参数
        config = {
            'file_path': 'D:/BaiduSyncdisk/课题1_RO Model/data_input.csv',
            'base_output_dir': 'D:/BaiduSyncdisk/课题1_RO Model/Results_XGB_20250122',
            'features': [
                'MWCO', 'contact_angle', 'pH', 'pres', 'initial_conc', 'meas_T', 
                'Total_Charge', 'Molecular_Weight', 'Molecular_Length', 
                'Molecular_Width', 'Molecular_Height', 'Dipole_moment', 
                'Polarizability', 'log_Kow'
            ],
            'target': 'rejection',
            
            #控制分组方式
            'group_methods': ['none', 'cmpd', 'ref'], #
            
            #控制特征工程方法
            'feature_methods': ['original']  # , 'standard', 'polynomial','interaction'
        }
        
        # 2. 创建实验管理器
        manager = ExperimentManager(config)
        
        # 3. 运行所有实验
        results = manager.run_all_experiments()
        
        # 4. 获取最佳实验
        best_exp = manager.logger.get_best_experiment()
        print("\n=== 最佳实验结果 ===")
        print(f"实验ID: {best_exp['experiment_id']}")
        print(f"分组方式: {best_exp['group_by']}")
        print(f"特征工程: {best_exp['feature_method']}")
        print(f"测试集 R2: {best_exp['test_r2']:.4f}")
        print(f"测试集 MAE: {best_exp['test_mae']:.4f}")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 