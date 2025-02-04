import json
import os

def check_feature_duplicates(config):
    """检查特征是否在多个机制组中重复出现"""
    feature_map = {}  # 记录每个特征属于哪些机制组
    
    for mechanism, group_info in config['mechanism_groups'].items():
        for feature in group_info['features']:
            if feature not in feature_map:
                feature_map[feature] = []
            feature_map[feature].append(mechanism)
    
    # 找出重复的特征
    duplicates = {
        feature: groups 
        for feature, groups in feature_map.items() 
        if len(groups) > 1
    }
    
    return duplicates

def validate_mechanism_config(config_path):
    """验证机制配置文件的格式和内容"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 验证基本结构
        if 'mechanism_groups' not in config:
            raise ValueError("配置文件缺少 'mechanism_groups' 键")
            
        # 验证每个机制组的结构
        for mechanism, group_info in config['mechanism_groups'].items():
            required_keys = ['color', 'features']
            if not all(key in group_info for key in required_keys):
                raise ValueError(f"机制组 '{mechanism}' 缺少必要的键: {required_keys}")
            
            # 验证颜色格式
            if not isinstance(group_info['color'], str) or not group_info['color'].startswith('#'):
                raise ValueError(f"机制组 '{mechanism}' 的颜色格式无效")
                
            # 验证特征列表
            if not isinstance(group_info['features'], list):
                raise ValueError(f"机制组 '{mechanism}' 的特征必须是列表")
        
        # 检查特征重复
        duplicates = check_feature_duplicates(config)
        if duplicates:
            print("警告：以下特征在多个机制组中重复出现：")
            for feature, groups in duplicates.items():
                print(f"  - {feature}: 出现在 {', '.join(groups)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"配置文件验证失败: {str(e)}")
        return False 