import xgboost as xgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from data_processor import apply_feature_engineering

def hyperparameter_optimization(X_train, y_train):
    """
    进行超参数优化。
    :param X_train: 训练集特征
    :param y_train: 训练集目标
    :return: 最佳模型
    """
    # 定义参数网格
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1],  # 更精细的学习率范围
        'n_estimators': [100, 200, 300, 500],  # 考虑更多的树数量
        'max_depth': [2, 3, 4, 5],  # 控制模型复杂度的树深度
        'min_child_weight': [1, 3, 5],  # 控制叶节点最小权重，可以防止过拟合
        'subsample': [0.6, 0.7, 0.8, 1.0],  # 控制数据的子采样比例，避免过拟合
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],  # 控制每棵树使用的特征子集比例
        'gamma': [0, 0.1, 0.3, 0.5],  # 增加节点分裂的惩罚，防止过拟合
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],  # L1 正则化力度
        'reg_lambda': [0.5, 1.0, 1.5, 2.0],  # L2 正则化力度
        'scale_pos_weight': [1]  # 对于类别不均衡问题，可以适当调整权重
    }

    # 初始化XGBoost回归模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 初始化随机搜索
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_grid, 
        n_iter=50,  # 随机搜索的迭代次数
        cv=3,       # 内部交叉验证折数
        verbose=1, 
        random_state=42, 
        n_jobs=-1,  # 使用所有CPU核心
        scoring='neg_mean_squared_error'
    )

    # 进行超参数搜索
    random_search.fit(X_train, y_train)
    print(f"最佳参数: {random_search.best_params_}")

    return random_search.best_estimator_

def train_and_evaluate_model_with_optimization(X_train, y_train, n_splits=5, feature_engineering_methods=['original', 'standard', 'polynomial', 'interaction'], feature_names=None):
    """
    使用5折交叉验证和超参数优化训练XGBoost模型并评估性能。
    :param X_train: 训练数据特征
    :param y_train: 训练数据目标值
    :param n_splits: 交叉验证折数
    :return: 训练完成的最佳XGBoost模型，交叉验证的评估指标和预测值
    """
    # 5折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储所有特征工程方法的结果
    results_dict = {}
    
    for method in feature_engineering_methods:
        print(f"\n评估特征工程方法: {method}")
        
        # 存储当前方法的评估指标
        method_results = {
            'mse_scores': [],
            'mae_scores': [],
            'r2_scores': [],
            'y_true': [],
            'y_pred': []
        }
        
        # 首先对整个训练集进行特征工程，获取完整的特征名列表
        X_train_full_processed, feature_info_full = apply_feature_engineering(X_train, method=method, feature_names=feature_names)
        full_feature_names = feature_info_full['feature_names']
    
        # 对每一折进行训练和评估
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"Training fold {fold_idx + 1}")
            
            # 分割训练集和验证集
            X_train_fold, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # 使用完整特征名列表应用特征工程
            X_train_processed, _ = apply_feature_engineering(X_train_fold, method=method, feature_names=full_feature_names)
            X_val_processed, _ = apply_feature_engineering(X_val, method=method, feature_names=full_feature_names)
            
            # 进行超参数优化
            best_model = hyperparameter_optimization(X_train_processed, y_train_fold)

            # 训练模型
            best_model.fit(
                X_train_processed,
                y_train_fold,
                early_stopping_rounds=10,
                eval_set=[(X_val_processed, y_val)],
                verbose=False
            )
            
            # 在验证集上进行预测
            y_pred = best_model.predict(X_val_processed)
            
            # 收集真实值和预测值
            method_results['y_true'].extend(y_val)
            method_results['y_pred'].extend(y_pred)
            
            # 计算评估指标
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            method_results['mse_scores'].append(mse)
            method_results['mae_scores'].append(mae)
            method_results['r2_scores'].append(r2)
            
            print(f"Fold {fold_idx + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # 计算平均评估指标
        avg_mse = np.mean(method_results['mse_scores'])
        std_mse = np.std(method_results['mse_scores'])
        avg_mae = np.mean(method_results['mae_scores'])
        std_mae = np.std(method_results['mae_scores'])
        avg_r2 = np.mean(method_results['r2_scores'])
        std_r2 = np.std(method_results['r2_scores'])
        
        print(f"\n{method} 方法评估指标:")
        print(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}")
        print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
        print(f"R2: {avg_r2:.4f} ± {std_r2:.4f}")
        
        # 存储该方法的所有结果
        results_dict[method] = {
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'detailed_results': method_results
        }
    
    return results_dict

def evaluate_on_test_data(model, X_test, y_test):
    """
    使用最终模型在测试集上进行评估。
    :param model: 训练完成的模型
    :param X_test: 测试集特征
    :param y_test: 测试集目标值
    :return: 测试集上的评估指标（MSE, MAE, R2）和预测值
    """
    # 测试集上的预测
    y_test_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n测试集评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    return mse, mae, r2, y_test, y_test_pred

if __name__ == "__main__":
    # 测试代码
    from data_processor import load_and_split_data, apply_feature_engineering, get_feature_columns
    
    # 加载数据
    file_path = 'D:/BaiduSyncdisk/课题1_RO Model/data_input.csv'
    features = get_feature_columns()
    target = 'rejection'
    
    # 测试不同分组方式
    for group_by in ['none', 'cmpd', 'ref']:
        print(f"\n测试 {group_by} 分组方式:")
        train_val_data, test_data = load_and_split_data(file_path, group_by=group_by)
        
        # 测试不同特征工程方法
        for method in ['original', 'standard', 'polynomial', 'interaction']:
            print(f"\n测试 {method} 特征工程方法:")
            
            # 应用特征工程
            X_train_processed, feature_info = apply_feature_engineering(
                train_val_data[features], 
                method=method
            )
            
            X_test_processed, _ = apply_feature_engineering(
                test_data[features],
                method=method,
                feature_names=feature_info['feature_names']
            )
            
            # 训练和评估模型
            model, avg_mse, avg_mae, avg_r2, y_train_true, y_train_pred = \
                train_and_evaluate_model_with_optimization(
                    X_train_processed,
                    train_val_data[target],
                    feature_names=feature_info['feature_names']
                )
            
            # 在测试集上评估
            test_mse, test_mae, test_r2, y_test, y_test_pred = \
                evaluate_on_test_data(
                    model,
                    X_test_processed, 
                    test_data[target]
                )