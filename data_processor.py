import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def load_and_split_data(file_path, group_by='none', test_size=0.2):
    """
    加载数据并按指定方式分割为训练/验证集和测试集。
    
    参数:
    file_path: 数据文件路径
    group_by: 分组方式，可选 'none'、'cmpd' 或 'ref'
    test_size: 测试集比例，默认0.2
    """
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 替换无效值，并删除目标列中含NaN的行
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['rejection'], inplace=True)

    if group_by not in ['none', 'cmpd', 'ref']:
        raise ValueError("group_by 必须是 'none'、'cmpd' 或 'ref'")

    if group_by == 'none':
        # 直接随机分割数据
        train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        print("不进行分组:")
        print(f"训练/验证集大小: {len(train_val_data)}")
        print(f"测试集大小: {len(test_data)}")
    else:
        # 按指定列分组
        unique_groups = data[group_by].unique()
        train_val_groups, test_groups = train_test_split(unique_groups, test_size=test_size, random_state=42)
        
        # 创建训练/验证集和测试集
        train_val_data = data[data[group_by].isin(train_val_groups)]
        test_data = data[data[group_by].isin(test_groups)]
        
        print(f"使用 {group_by} 进行分组:")
        print(f"训练/验证集大小: {len(train_val_data)}")
        print(f"测试集大小: {len(test_data)}")
        print(f"训练/验证集中的{group_by}数量: {len(train_val_groups)}")
        print(f"测试集中的{group_by}数量: {len(test_groups)}")
    
    return train_val_data, test_data

def apply_feature_engineering(data, method='standard', feature_names=None):
    """
    应用特征工程
    
    参数:
    data: 原始数据
    method: 特征工程方法，可选 'original'、'standard'、'polynomial'、'interaction'
    feature_names: 如果提供，将使用这些特征名进行处理
    
    返回:
    transformed_data: 处理后的数据
    feature_info: 特征工程信息字典
    """
    feature_info = {
        'method': method,
        'features_used': [],
        'transformations': [],
        'scaler': None,  # 保存标准化器
        'poly': None,    # 保存多项式特征生成器
        'feature_names': None  # 保存特征名列表
    }
    
    if feature_names is not None:
        data = data[feature_names]
    
    if method == 'original':
        transformed_data = data.copy()
        feature_info['features_used'] = list(data.columns)
        feature_info['transformations'].append('original')
        feature_info['feature_names'] = list(data.columns)
        
    elif method == 'standard':
        # 标准化处理
        scaler = StandardScaler()
        transformed_data = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        feature_info['scaler'] = scaler
        feature_info['transformations'].append('StandardScaler')
        feature_info['features_used'] = list(data.columns)
        feature_info['feature_names'] = list(data.columns)
        
    elif method == 'polynomial':
        # 多项式特征
        poly = PolynomialFeatures(degree=2, include_bias=False)
        transformed_array = poly.fit_transform(data)
        # 生成新的特征名
        feature_names = poly.get_feature_names_out(data.columns)
        transformed_data = pd.DataFrame(
            transformed_array,
            columns=feature_names,
            index=data.index
        )
        feature_info['poly'] = poly
        feature_info['transformations'].append('PolynomialFeatures_degree2')
        feature_info['features_used'] = list(feature_names)
        feature_info['feature_names'] = list(feature_names)
        
    elif method == 'interaction':
        transformed_data = data.copy()
        
        # 定义物理化学机制相关的特征组
        interactions = {
            '静电作用': ['Total_Charge', 'pH', 'Dipole_moment'],
            '位阻效应': ['Molecular_Length', 'Molecular_Width', 'Molecular_Height', 'MWCO'],
            '分子间作用力': ['Polarizability', 'log_Kow', 'contact_angle']
        }
        
        # 记录所有交互特征名
        interaction_features = []
        
        # 组内交互
        for mechanism, features in interactions.items():
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    if features[i] in data.columns and features[j] in data.columns:
                        col_name = f"{features[i]}_{features[j]}_{mechanism}"
                        transformed_data[col_name] = data[features[i]] * data[features[j]]
                        feature_info['transformations'].append(f"Interaction_{mechanism}_{col_name}")
                        interaction_features.append(col_name)
        
        # 组间关键交互
        key_cross_interactions = [
            ('Total_Charge', 'Molecular_Length'),  # 静电作用与位阻的协同效应
            ('log_Kow', 'MWCO'),                  # 疏水性与孔径的关系
            ('Dipole_moment', 'contact_angle'),    # 极性与表面性质的关系
            ('pH', 'Molecular_Width'),             # pH影响下的位阻效应
        ]
        
        for feat1, feat2 in key_cross_interactions:
            if feat1 in data.columns and feat2 in data.columns:
                col_name = f"{feat1}_{feat2}_cross_interaction"
                transformed_data[col_name] = data[feat1] * data[features[j]]
                feature_info['transformations'].append(f"CrossInteraction_{col_name}")
                interaction_features.append(col_name)
        
        # 保存所有特征名（原始特征 + 交互特征）
        all_features = list(data.columns) + interaction_features
        feature_info['features_used'] = all_features
        feature_info['feature_names'] = all_features
    
    else:
        raise ValueError("不支持的特征工程方法")
    
    return transformed_data, feature_info

def get_feature_columns():
    """
    返回特征列名列表
    """
    return ['MWCO', 'contact_angle', 'pH', 'pres', 'initial_conc', 'meas_T', 
            'Total_Charge', 'Molecular_Weight', 'Molecular_Length', 
            'Molecular_Width', 'Molecular_Height', 'Dipole_moment', 
            'Polarizability', 'log_Kow']

if __name__ == "__main__":
    # 测试代码
    file_path = 'D:/BaiduSyncdisk/课题1_RO Model/data_input.csv'
    features = get_feature_columns()
    
    # 测试不同分组方式
    for group_by in ['none', 'cmpd']: # , 'ref'
        print(f"\n测试 {group_by} 分组方式:")
        train_val_data, test_data = load_and_split_data(file_path, group_by=group_by)
        
        # 测试特征工程
        for method in ['original', 'standard', 'interaction']: #'polynomial',
            print(f"\n测试 {method} 特征工程方法:")
            X_train_processed, feature_info = apply_feature_engineering(
                train_val_data[features], 
                method=method
            )
            print(f"处理后的特征数量: {len(feature_info['features_used'])}")
            print(f"应用的转换: {feature_info['transformations']}") 