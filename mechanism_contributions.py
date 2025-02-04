import numpy as np
import pandas as pd
import shap
import json
import os
from scipy.stats import spearmanr

class MechanismAnalyzer:
    """机制贡献分析器 - 用于分析和量化不同物理化学机制对分离过程的影响"""
    
    def __init__(self, feature_groups=None):
        """
        初始化分析器
        
        参数:
        feature_groups: dict, 特征组字典，定义各机制包含的特征
        """
        if feature_groups is None:
            # 从配置文件读取
            config_path = os.path.join(os.path.dirname(__file__), 'mechanism_config.json')
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # 提取特征组信息
                self.feature_groups = {
                    mechanism: group_info['features']
                    for mechanism, group_info in config['mechanism_groups'].items()
                }
            except Exception as e:
                print(f"读取配置文件失败: {str(e)}")
                # 使用默认配置
                self.feature_groups = {
                    'Electrostatic': ['Total_Charge', 'pH', 'Dipole_moment'],
                    'Steric': ['MWCO', 'Molecular_Weight', 'Molecular_Length', 
                              'Molecular_Width', 'Molecular_Height'],
                    'van der Waals': ['Polarizability', 'Molecular_Weight'],
                    'Hydrophobic': ['log_Kow', 'contact_angle']
                }
        else:
            self.feature_groups = feature_groups
        
        # 添加特征验证
        self._validate_features()
    
    def _validate_features(self):
        """验证特征组是否有重复"""
        all_features = []
        feature_mechanisms = {}
        
        for mechanism, features in self.feature_groups.items():
            for feature in features:
                if feature not in feature_mechanisms:
                    feature_mechanisms[feature] = []
                feature_mechanisms[feature].append(mechanism)
                all_features.append(feature)
        
        # 检查重复特征
        duplicates = {
            feature: mechanisms 
            for feature, mechanisms in feature_mechanisms.items() 
            if len(mechanisms) > 1
        }
        
        if duplicates:
            print("警告：以下特征在多个机制组中重复出现：")
            for feature, mechanisms in duplicates.items():
                print(f"  - {feature}: 出现在 {', '.join(mechanisms)}")
                # 默认将重复特征只分配给第一个机制组
                first_mechanism = mechanisms[0]
                for other_mechanism in mechanisms[1:]:
                    self.feature_groups[other_mechanism].remove(feature)
                print(f"    已自动将 {feature} 只分配给 {first_mechanism} 机制组")
    
    def analyze(self, model, data):
        """执行完整的机制分析"""
        try:
            # 1. 计算SHAP值
            shap_values = self._calculate_shap_values(model, data)
            
            # 2. 获取特征重要性
            feature_importance = self._get_feature_importance(model, data)
            
            # 3. 分析各机制贡献
            mechanism_contributions = self._analyze_mechanisms(
                data, shap_values, feature_importance
            )
            
            # 4. 分析机制间交互
            mechanism_interactions = self._analyze_interactions(
                data, shap_values
            )
            
            return {
                'mechanism_contributions': mechanism_contributions,
                'mechanism_interactions': mechanism_interactions,
                'raw_shap_values': shap_values,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            print(f"分析过程发生错误: {str(e)}")
            return None
    
    def _calculate_shap_values(self, model, data):
        """计算SHAP值"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)
            return shap_values[0] if isinstance(shap_values, list) else shap_values
        except Exception as e:
            print(f"SHAP值计算失败: {str(e)}")
            return np.zeros((data.shape[0], data.shape[1]))
    
    def _get_feature_importance(self, model, data):
        """获取特征重要性"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(data.shape[1])
            return pd.Series(importance, index=data.columns)
        except Exception as e:
            print(f"特征重要性计算失败: {str(e)}")
            return pd.Series(np.zeros(data.shape[1]), index=data.columns)
    
    def _analyze_mechanisms(self, data, shap_values, feature_importance):
        """分析各机制的贡献度"""
        mechanism_contributions = {}
        
        for mechanism, features in self.feature_groups.items():
            try:
                valid_features = [f for f in features if f in data.columns]
                if not valid_features:
                    continue
                
                feature_indices = [data.columns.get_loc(f) for f in valid_features]
                
                # 计算SHAP重要性并处理异常值
                shap_importance = np.abs(shap_values[:, feature_indices]).mean(axis=0)
                shap_importance = np.clip(shap_importance, 0, None)
                shap_importance = np.nan_to_num(shap_importance, 0)
                
                # 获取特征重要性并标准化
                feature_imp = feature_importance[valid_features].values
                feature_imp = np.clip(feature_imp, 0, None)
                feature_imp = np.nan_to_num(feature_imp, 0)
                
                # 计算特征间相关性
                corr_matrix = data[valid_features].corr().abs().values
                corr_matrix = np.nan_to_num(corr_matrix, 0)
                
                # 计算综合得分
                total_score = (0.5 * np.mean(shap_importance) + 
                             0.3 * np.mean(feature_imp) +
                             0.2 * np.mean(corr_matrix))
                
                mechanism_contributions[mechanism] = {
                    'shap_contribution': float(np.mean(shap_importance)),
                    'feature_importance': float(np.mean(feature_imp)),
                    'feature_correlation': float(np.mean(corr_matrix)),
                    'features': valid_features,
                    'total_score': float(total_score)
                }
                
            except Exception as e:
                print(f"分析机制 {mechanism} 时出错: {str(e)}")
                mechanism_contributions[mechanism] = self._get_default_contribution(valid_features)
        
        return mechanism_contributions
    
    def _analyze_interactions(self, data, shap_values):
        """分析机制间的交互作用"""
        interactions = {}
        
        for mech1, features1 in self.feature_groups.items():
            for mech2, features2 in self.feature_groups.items():
                if mech1 >= mech2:
                    continue
                
                valid_features1 = [f for f in features1 if f in data.columns]
                valid_features2 = [f for f in features2 if f in data.columns]
                
                if not valid_features1 or not valid_features2:
                    continue
                
                interaction_strength = self._calculate_interaction_strength(
                    data, shap_values, valid_features1, valid_features2
                )
                
                interactions[f"{mech1}-{mech2}"] = float(interaction_strength)
        
        return interactions
    
    def _calculate_interaction_strength(self, data, shap_values, features1, features2):
        """计算两组特征间的交互强度"""
        try:
            idx1 = [data.columns.get_loc(f) for f in features1]
            idx2 = [data.columns.get_loc(f) for f in features2]
            
            shap_correlations = []
            feature_correlations = []
            
            # 计算SHAP值相关性
            for i in idx1:
                shap_values_i = shap_values[:, i]
                for j in idx2:
                    shap_values_j = shap_values[:, j]
                    if len(np.unique(shap_values_i)) > 1 and len(np.unique(shap_values_j)) > 1:
                        try:
                            corr = abs(spearmanr(shap_values_i, shap_values_j)[0])
                            if not np.isnan(corr):
                                shap_correlations.append(corr)
                        except:
                            continue
            
            # 计算特征相关性
            for f1 in features1:
                values1 = data[f1].values
                for f2 in features2:
                    values2 = data[f2].values
                    if len(np.unique(values1)) > 1 and len(np.unique(values2)) > 1:
                        try:
                            corr = abs(spearmanr(values1, values2)[0])
                            if not np.isnan(corr):
                                feature_correlations.append(corr)
                        except:
                            continue
            
            shap_corr = np.mean(shap_correlations) if shap_correlations else 0
            feature_corr = np.mean(feature_correlations) if feature_correlations else 0
            
            return float(0.7 * shap_corr + 0.3 * feature_corr)
            
        except Exception as e:
            print(f"计算交互强度时出错: {str(e)}")
            return 0.0
    
    def _get_default_contribution(self, features):
        """返回默认的贡献值"""
        return {
            'shap_contribution': 0.0,
            'feature_importance': 0.0,
            'feature_correlation': 0.0,
            'features': features,
            'total_score': 0.0
        }

def run_analysis(model, data, output_dir):
    """运行完整的机制分析流程"""
    analyzer = MechanismAnalyzer()
    
    # 执行分析
    results = analyzer.analyze(model, data)
    
    # 生成报告
    report = {
        'mechanism_scores': {
            mechanism: {
                'total_score': details['total_score'],
                'shap_contribution': details['shap_contribution'],
                'feature_importance': details['feature_importance'],
                'feature_correlation': details['feature_correlation'],
                'features': details['features']
            }
            for mechanism, details in results['mechanism_contributions'].items()
        },
        'mechanism_interactions': results['mechanism_interactions']
    }
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'mechanism_analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return results, report