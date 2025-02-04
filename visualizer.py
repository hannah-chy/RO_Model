import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置后端
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import shap
import os
import networkx as nx
from matplotlib.collections import LineCollection
import json

# 设置全局可视化参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class Visualizer:
    """统一的可视化类"""
    
    def __init__(self, output_dir):
        # 在初始化时设置后端
        matplotlib.use('Agg')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取配置文件
        config_path = os.path.join(os.path.dirname(__file__), 'mechanism_config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"读取配置文件失败: {str(e)}")
            self.config = None
        
    def plot_model_results(self, model, X_test, feature_names, metrics):
        """绘制模型预测结果"""
        self._plot_scatter(metrics)
        self._plot_shap_summary(model, X_test, feature_names)
    
    def plot_mechanism_analysis(self, results):
        """绘制机制分析结果"""
        # 添加数据结构验证
        required_keys = ['mechanism_contributions', 'mechanism_interactions']
        if not all(key in results for key in required_keys):
            raise ValueError(f"结果数据缺少必要的键: {required_keys}")
        
        self._plot_mechanism_contributions(results['mechanism_contributions'])
        self._plot_mechanism_interactions(results['mechanism_interactions'])
        self._plot_mechanism_network(results['mechanism_interactions'])
        
    def _plot_scatter(self, metrics):
        """绘制预测值vs实际值散点图"""
        try:
            plt.figure(figsize=(8.5, 8))
            
            # 设置背景样式
            plt.style.use('default')
            ax = plt.gca()
            ax.set_facecolor('white')
            ax.grid(False)
            
            # 绘制散点图
            plt.scatter(metrics['y_train_true'], metrics['y_train_pred'], 
                       color='#EDAE92', alpha=0.6, s=150,
                       label=f'Train (MAE = {metrics["train_mae"]:.2f})')
            plt.scatter(metrics['y_test_true'], metrics['y_test_pred'], 
                       color='#6C96CC', alpha=0.6, s=150, 
                       label=f'Test (MAE = {metrics["test_mae"]:.2f})')
            
            # 绘制 y=x 参考线
            x = np.linspace(0, max(metrics['y_test_true']), 100)
            plt.plot(x, x, '--', color='black', label='y=x', linewidth=3)
            
            # 设置标签和图例
            plt.xlabel('Real Values', fontsize=24, fontname='Arial')
            plt.ylabel('Predicted Values', fontsize=24, fontname='Arial')
            plt.legend(prop={'family': 'Arial', 'size': 22})
            
            # 设置边框和刻度
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(3)
            ax.tick_params(axis='both', direction='in', labelsize=22)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'scatter_plot.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制散点图时出错: {str(e)}")
    
    def _plot_shap_summary(self, model, X_test, feature_names):
        """绘制SHAP值摘要图"""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # 创建自定义颜色映射
            custom_cmap = LinearSegmentedColormap.from_list('custom', ['#EDAE92', '#6C96CC'])
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values if isinstance(shap_values, np.ndarray) else shap_values[0],
                X_test,
                feature_names=feature_names,
                show=False,
                plot_size=(12, 8),
                alpha=0.8,
                max_display=15,
                color_bar_label='Feature Value',
                cmap=custom_cmap  # 使用创建的颜色映射
            )
            
            plt.title('Feature Importance Analysis (SHAP Values)', 
                     fontsize=24, pad=20)
            plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=20)
            
            ax = plt.gca()
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制SHAP摘要图时出错: {str(e)}")
    
    def _plot_mechanism_contributions(self, contributions):
        """绘制机制贡献饼图"""
        try:
            plt.figure(figsize=(12, 12))
            
            # 使用配置文件中的机制组定义
            if self.config and 'mechanism_groups' in self.config:
                mechanism_groups = self.config['mechanism_groups']
            else:
                # 使用默认配置
                mechanism_groups = {
                        'Electrostatic': {
                            'color': '#6C96CC',
                            'features': ['Total_Charge', 'pH', 'Dipole_moment']
                        },
                        'Steric': {
                        'color': '#A4C8D9',
                            'features': ['MWCO', 'Molecular_Weight', 'Molecular_Length', 
                                    'Molecular_Width', 'Molecular_Height']
                        },
                        'van der Waals': {
                        'color': '#EDAE92',
                            'features': ['Polarizability', 'Molecular_Weight']
                    },
                        'Hydrophobic': {
                        'color': '#C92321',
                            'features': ['log_Kow', 'contact_angle']
                        }
                    }
            
            # 计算机制总分和特征总分
            mechanism_scores = {}
            feature_scores = {}
            total_score = 0
            
            # 1. 首先计算每个特征的得分
            for mech, details in contributions.items():
                score = (0.7 * details['shap_contribution'] + 
                        0.3 * details['feature_importance'])
                mechanism_scores[mech] = score
                total_score += score
                
                # 计算每个特征的得分
                n_features = len(details['features'])
                if n_features > 0:
                    feature_score = score / n_features  # 平均分配给每个特征
                    for feature in details['features']:
                        feature_scores[feature] = feature_score

            # 2. 准备饼图数据
            inner_labels = list(mechanism_groups.keys())
            inner_values = []
            outer_labels = []
            outer_values = []
            outer_colors = []
            
            # 3. 计算内圈（机制）数据
            for mechanism in inner_labels:
                if mechanism in mechanism_scores:
                    value = (mechanism_scores[mechanism] / total_score) * 100
                    inner_values.append(value)
                else:
                    inner_values.append(0)

            # 4. 计算外圈（特征）数据
            for mechanism, group_info in mechanism_groups.items():
                color = group_info['color']
                for feature in group_info['features']:
                    if feature in feature_scores:
                        outer_labels.append(feature)
                        value = (feature_scores[feature] / total_score) * 100
                        outer_values.append(value)
                        outer_colors.append(color)

            # 5. 绘制饼图
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # 内圈（机制）
            inner_colors = [group_info['color'] for group_info in mechanism_groups.values()]
            ax.pie(inner_values, radius=0.6, labels=None, colors=inner_colors,
                   autopct='%1.1f%%', pctdistance=0.85,
                   wedgeprops=dict(width=0.5, edgecolor='white'),
                   labeldistance=0.2,
                   textprops={'fontsize': 18, 'fontweight': 'bold'})
            
            # 外圈（特征）
            wedges, texts = ax.pie(outer_values, radius=1.2, labels=outer_labels, colors=outer_colors,
                   autopct=None, pctdistance=0.85,
                   wedgeprops=dict(width=0.5, edgecolor='white'),
                    labeldistance=0.6,
                   textprops={'fontsize': 14, 'fontweight': 'bold'},
                   rotatelabels=True)
            
            # 添加图例
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_info['color'],
                                           label=mechanism)
                             for mechanism, group_info in mechanism_groups.items()]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

            plt.title('Mechanism Contributions', fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'mechanism_contributions.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制机制贡献饼图时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_mechanism_interactions(self, interactions):
        """绘制机制交互热图"""
        try:
            # 获取所有唯一的机制名称
            mechanisms = sorted(list(set([m.split('-')[0] for m in interactions.keys()] + 
                                    [m.split('-')[1] for m in interactions.keys()])))
            
            # 创建交互矩阵
            n = len(mechanisms)
            matrix = np.zeros((n, n))
            
            # 填充矩阵
            for pair, strength in interactions.items():
                mech1, mech2 = pair.split('-')
                i = mechanisms.index(mech1)
                j = mechanisms.index(mech2)
                matrix[i, j] = float(strength)
                matrix[j, i] = float(strength)  # 对称矩阵
            
            # 创建图形
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            
            # 设置背景色和网格
            ax.set_facecolor('white')
            
            # 使用seaborn绘制热图
            sns.heatmap(matrix,
                    xticklabels=mechanisms,
                    yticklabels=mechanisms,
                       cmap='RdBu_r',  # 红蓝配色
                       center=0,
                       annot=True,  # 显示数值
                       fmt='.3f',   # 数值格式（3位小数）
                       square=True,  # 正方形单元格
                       cbar_kws={'label': 'Interaction Strength'},
                       mask=np.triu(np.ones_like(matrix, dtype=bool)),  # 只显示下三角
                       )
            
            # 设置标题和样式
            plt.title('Mechanism Interactions', fontsize=14, pad=20)
            
            # 调整刻度标签
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(os.path.join(self.output_dir, 'mechanism_heatmap.png'),
                       dpi=300,
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制机制交互热图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _plot_mechanism_network(self, interactions, threshold=0.1):
        """绘制机制交互网络图"""
        try:
            # 设置样式
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 10), dpi=300)
            gs = fig.add_gridspec(1, 20)
            ax = fig.add_subplot(gs[0, :19])
            cax = fig.add_subplot(gs[0, 19:])
            
            # 设置背景
            ax.set_facecolor('#F8F9FA')
            ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
            
            # 创建无向图
            G = nx.Graph()
            
            # 添加边和节点属性
            max_weight = max(float(strength) for strength in interactions.values())
            
            # 计算每个节点的总强度
            node_strengths = {}
            for pair, strength in interactions.items():
                mech1, mech2 = pair.split('-')
                strength = float(strength)  # 确保强度是浮点数
                
                # 更新节点强度
                for node in [mech1, mech2]:
                    if node not in node_strengths:
                        node_strengths[node] = 0.0
                    node_strengths[node] += strength
                
                # 只添加超过阈值的边
                if strength >= threshold:
                    G.add_edge(mech1, mech2, weight=strength)
            
            if len(G.edges()) == 0:
                print(f"警告：没有强度大于{threshold}的交互")
                return
                
            # 使用spring_layout以获得更好的节点分布
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 计算节点大小
            max_strength = max(node_strengths.values())
            node_sizes = {
                node: (strength / max_strength) * 3000 + 1000  # 确保最小尺寸
                for node, strength in node_strengths.items()
            }
            
            # 获取边的权重和颜色
            edges = G.edges()
            weights = [float(G[u][v]['weight']) for u, v in edges]
            
            # 使用自定义颜色映射
            edge_cmap = LinearSegmentedColormap.from_list('custom', ['#6C96CC', '#EDAE92', '#C92321'])
            edge_colors = edge_cmap(np.array(weights) / max_weight)
            
            # 创建渐变边
            for (u, v), color in zip(edges, edge_colors):
                x = np.linspace(pos[u][0], pos[v][0], 100)
                y = np.linspace(pos[u][1], pos[v][1], 100)
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=[color], 
                                  linewidths=float(G[u][v]['weight']) * 5,
                                alpha=0.7, zorder=1)
                ax.add_collection(lc)
            
            # 绘制节点光晕
            for node, (x, y) in pos.items():
                size = node_sizes[node]
                for i, alpha in [(1.3, 0.1), (1.1, 0.2)]:
                    circle = plt.Circle((x, y), 
                                    radius=np.sqrt(size) / 180,
                                    color='#EDAE92', 
                                    alpha=alpha, 
                                    zorder=1)
                    ax.add_patch(circle)
            
            # 绘制节点
            nodes = nx.draw_networkx_nodes(G, pos,
                                        node_color='#6C96CC',
                                        node_size=[node_sizes[node] for node in G.nodes()],
                                        alpha=0.8,
                                        linewidths=2,
                                        edgecolors='white',
                                        node_shape='o',
                                        ax=ax)
            
            # 手动设置节点的zorder
            if hasattr(nodes, 'set_zorder'):
                nodes.set_zorder(2)  # 节点在光晕之上
            
            # 添加节点标签（机制名称）
            labels = {node: node for node in G.nodes()}  # 使用机制名称作为标签
            label_nodes = nx.draw_networkx_labels(G, pos,
                                          labels=labels,
                                          font_size=12,
                                        font_weight='bold',
                                        font_family='Arial',
                                        bbox=dict(facecolor='white',
                                                edgecolor='none',
                                                  alpha=0.9,
                                                pad=0.5))
            
            # 手动设置标签的zorder
            for t in label_nodes.values():
                if hasattr(t, 'set_zorder'):
                    t.set_zorder(3)  # 标签在最上层
            
            # 添加边标签（交互强度）
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f'{float(v):.3f}' for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels=edge_labels,
                                    font_size=10,
                                    font_weight='bold',
                                    bbox=dict(facecolor='white',
                                            edgecolor='none',
                                            alpha=0.9,
                                            pad=0.3),
                                    label_pos=0.5,  # 标签位置在边的中间
                                    rotate=False)   # 不旋转标签
            
            # 调整节点位置以避免标签重叠
            pos_labels = {}
            for node in G.nodes():
                pos_labels[node] = pos[node] + np.array([0, 0.12])  # 略微向上偏移
            
            # 添加节点强度标签和文本名称标签
            for node, (x, y) in pos.items():
                strength = node_strengths[node]
                # 添加强度标签
                ax.text(x, y - 0.15,  # 在节点下方添加强度值
                       f'Strength: {strength:.3f}',
                       horizontalalignment='center',
                       verticalalignment='top',
                       fontsize=8,
                       fontweight='bold',
                       bbox=dict(facecolor='white',
                               edgecolor='none',
                               alpha=0.8,
                               pad=0.3))
                               
                # 添加文本名称标签
                if self.config and 'mechanism_names' in self.config:
                    text_name = self.config['mechanism_names'].get(node, node)
                    ax.text(x, y + 0.15,  # 在节点上方添加文本名称
                           text_name,
                           horizontalalignment='center',
                           verticalalignment='bottom',
                           fontsize=10,
                           fontweight='bold',
                           bbox=dict(facecolor='white',
                                   edgecolor='none',
                                   alpha=0.8,
                                   pad=0.3))
            
            # 设置标题
            ax.set_title('Mechanism Interaction Network', 
                        fontsize=16, 
                        fontweight='bold',
                        pad=20,
                        fontfamily='Arial')
            ax.axis('off')
            
            # 添加颜色条
            norm = plt.Normalize(vmin=min(weights), vmax=max_weight)
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Interaction Strength', 
                        fontsize=12,
                        fontweight='bold',
                        fontfamily='Arial')
            
            # 添加图例
            strength_range = np.array([min(node_strengths.values()), 
                                    max(node_strengths.values())])
            legend_sizes = (strength_range / max_strength) * 3000 + 1000
            legend_labels = [f'Weak: {strength_range[0]:.2f}',
                            f'Strong: {strength_range[1]:.2f}']
            
            legend_elements = [
                plt.scatter([], [], 
                                        s=size,
                                        c='#6C96CC',
                                        alpha=0.8,
                        label=f'{label}\nStrength: {strength:.3f}')
                for size, (label, strength) in zip(legend_sizes, 
                                               zip(['Weak', 'Strong'], strength_range))]
            
            ax.legend(handles=legend_elements,
                    loc='upper left',
                    bbox_to_anchor=(1.1, 0.2),
                    fontsize=10,
                    title='Node Strength',
                    title_fontsize=11)
            
            # 保存图片时增加边距
            plt.savefig(os.path.join(self.output_dir, 'mechanism_network.png'),
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.2)  # 增加边距
            plt.close()
            
        except Exception as e:
            print(f"绘制机制网络图时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈