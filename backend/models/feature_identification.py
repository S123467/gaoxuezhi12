"""
模块一：高血脂关键风险因素识别
算法：Lasso-Logistic 回归、XGBoost-SHAP、随机森林交叉验证
论文对应：问题一 - 从多维度特征中筛选关键影响因素

论文结论：
- TG、TC、LDL-C、HDL-C 及血尿酸构成高血脂风险识别的主要代谢主轴
- 痰湿质、BMI 及 ADL/IADL 等变量在非线性依赖与交互分析中表现出重要补充价值
- 痰湿质与 BMI 的交互放大效应尤为显著

Author: 基于论文CMC2604725研究成果
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional


class KeyRiskFactorIdentifier:
    """
    高血脂关键风险因素识别器
    实现论文中的三步法：
    1. Lasso-Logistic 回归进行特征筛选和线性主效应推断
    2. XGBoost-SHAP 挖掘非线性特征重要性和交互机制
    3. 随机森林交叉验证进行多模型交叉验证
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.lasso = Lasso(alpha=0.01, random_state=random_state)
        self.logistic = LogisticRegression(penalty='l2', random_state=random_state)
        self.xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.explainer = None
        self.is_fitted = False
        
        # 论文确定的核心特征名称
        self.expected_core_features = [
            'tg', 'tc', 'ldl_c', 'hdl_c', 'urea_acid', 
            'tan_score', 'bmi', 'activity_score'
        ]
        
    def fit_lasso_logistic(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict:
        """
        第一步：Lasso-Logistic 特征筛选和线性主效应推断
        
        参数:
            X: 特征矩阵
            y: 目标标签 (0=正常, 1=高血脂)
            
        返回:
            筛选结果字典
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # Lasso 特征筛选
        self.lasso.fit(X_scaled, y)
        selector = SelectFromModel(self.lasso, prefit=True)
        X_lasso = selector.transform(X_scaled)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Logistic 回归得到系数和 OR 值
        self.logistic.fit(X_lasso, y)
        # 计算 OR 值 (exp(系数))
        coef = self.logistic.coef_[0]
        or_values = np.exp(coef)
        
        # 按 OR 绝对值排序
        feature_coef = list(zip(selected_features, coef, or_values))
        feature_coef.sort(key=lambda x: abs(x[1]), reverse=True)
        
        result = {
            'method': 'Lasso-Logistic',
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'feature_coef': feature_coef,
            'X_lasso_shape': X_lasso.shape
        }
        
        print(f"Lasso-Logistic 筛选完成，选出 {len(selected_features)} 个特征")
        for name, c, or_val in feature_coef[:10]:
            print(f"  {name}: 系数={c:.3f}, OR={or_val:.3f}")
        
        return result
    
    def fit_xgboost_shap(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict:
        """
        第二步：XGBoost-SHAP 非线性特征重要性分析
        
        参数:
            X: 特征矩阵
            y: 目标标签
            
        返回:
            SHAP分析结果
        """
        X_scaled = self.scaler.fit_transform(X)
        self.xgb.fit(X_scaled, y)
        
        # 初始化SHAP解释器
        self.explainer = shap.TreeExplainer(self.xgb)
        shap_values = self.explainer.shap_values(X_scaled)
        
        # 计算全局特征重要性（平均绝对SHAP值）
        feature_importance = []
        for i, feature in enumerate(X.columns):
            importance = np.mean(np.abs(shap_values[:, i]))
            feature_importance.append((feature, importance))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 计算特征贡献方向
        # SHAP值为正表示增加风险，负表示降低风险
        direction = []
        for i, feature in enumerate(X.columns):
            mean_shap = np.mean(shap_values[:, i])
            direction.append((feature, mean_shap, "增加风险" if mean_shap > 0 else "降低风险"))
        
        result = {
            'method': 'XGBoost-SHAP',
            'feature_importance': feature_importance,
            'shap_direction': direction,
            'shap_values': shap_values
        }
        
        print("\nXGBoost-SHAP 特征重要性（Top 10）:")
        for feature, imp in feature_importance[:10]:
            print(f"  {feature}: {imp:.4f}")
        
        return result
    
    def cross_validate_random_forest(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict:
        """
        第三步：随机森林交叉验证，进行多模型一致性验证
        
        参数:
            X: 特征矩阵
            y: 目标标签
            cv_folds: 交叉验证折数
            
        返回:
            验证结果
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # 5折分层交叉验证
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.rf, X_scaled, y, cv=skf, scoring='accuracy')
        
        # 在全量数据上训练得到特征重要性
        self.rf.fit(X_scaled, y)
        rf_importance = self.rf.feature_importances_
        feature_importance = list(zip(X.columns, rf_importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'method': 'RandomForest-CV',
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'feature_importance': feature_importance
        }
        
        print(f"\n随机森林 {cv_folds} 折交叉验证:")
        print(f"  平均准确率: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        print("\n随机森林特征重要性（Top 10）:")
        for feature, imp in feature_importance[:10]:
            print(f"  {feature}: {imp:.4f}")
        
        return result
    
    def fit_all(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict:
        """
        执行完整的三步特征识别流程
        
        参数:
            X: 特征矩阵
            y: 目标标签
            
        返回:
            完整结果字典
        """
        result_lasso = self.fit_lasso_logistic(X, y)
        result_xgb = self.fit_xgboost_shap(X, y)
        result_rf = self.cross_validate_random_forest(X, y)
        
        # 综合三个模型，得到共识核心特征
        # 取三个模型都排在前10的特征
        lasso_top = set([f[0] for f in result_lasso['feature_coef'][:10]])
        xgb_top = set([f[0] for f in result_xgb['feature_importance'][:10]])
        rf_top = set([f[0] for f in result_rf['feature_importance'][:10]])
        
        consensus_features = list(lasso_top & xgb_top & rf_top)
        
        # 检查是否与论文结论一致
        paper_core = ['tg', 'tc', 'ldl_c', 'hdl_c', 'urea_acid', 'tan_score', 'bmi']
        found_paper_core = [f for f in paper_core if f in consensus_features]
        
        result = {
            'lasso_result': result_lasso,
            'xgb_result': result_xgb,
            'rf_result': result_rf,
            'consensus_features': consensus_features,
            'n_consensus': len(consensus_features),
            'paper_core_found': found_paper_core,
            'n_paper_core_found': len(found_paper_core)
        }
        
        self.is_fitted = True
        
        print(f"\n===== 综合结论 =====")
        print(f"三个模型共识核心特征 ({len(consensus_features)} 个): {consensus_features}")
        print(f"论文核心特征命中: {len(found_paper_core)}/{len(paper_core)}")
        
        return result
    
    def get_feature_weight_summary(self, result: Dict) -> pd.DataFrame:
        """
        从结果中提取特征权重汇总表
        
        参数:
            result: fit_all 返回的结果
            
        返回:
            特征权重汇总DataFrame
        """
        summary = []
        
        # 从Lasso获取
        for name, coef, or_val in result['lasso_result']['feature_coef']:
            summary.append({
                'feature': name,
                'lasso_coef': coef,
                'lasso_or': or_val
            })
        
        df_lasso = pd.DataFrame(summary)
        
        # 加入XGBoost重要性
        xgb_map = {f: imp for f, imp in result['xgb_result']['feature_importance']}
        df_lasso['xgb_importance'] = df_lasso['feature'].map(xgb_map)
        
        # 加入RF重要性
        rf_map = {f: imp for f, imp in result['rf_result']['feature_importance']}
        df_lasso['rf_importance'] = df_lasso['feature'].map(rf_map)
        
        # 排序
        df_lasso = df_lasso.sort_values('xgb_importance', ascending=False)
        
        return df_lanno
    
    def plot_feature_importance(
        self, 
        result: Dict, 
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """绘制特征重要性对比图"""
        importance = result['xgb_result']['feature_importance'][:top_n]
        features = [f[0] for f in importance]
        values = [f[1] for f in importance]
        features.reverse()
        values.reverse()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, values)
        ax.set_xlabel('平均绝对SHAP值 (特征重要性)')
        ax.set_title('XGBoost-SHAP 特征重要性 (Top {top_n})')
        
        # 高亮论文核心特征
        paper_core = ['tg', 'tc', 'ldl_c', 'hdl_c', 'urea_acid', 'tan_score', 'bmi']
        for i, feature in enumerate(features):
            if feature in paper_core:
                bars[i].set_color('#d62728')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    # 简单测试：使用模拟数据验证流程
    print("=== 高血脂关键风险因素识别 - 测试 ===")
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[
            'age', 'bmi', 'tg', 'tc', 'hdl_c', 'ldl_c', 'urea_acid', 
            'tan_score', 'qi_xu', 'yang_xu', 'yin_xu', 'yang_zhi',
            'tan_yu', 'shi_re', 'xue_yu', 'qi_zhi', 'feng_re',
            'activity_score', 'gender', 'waist'
        ]
    )
    
    # 让目标与核心特征相关（模拟真实数据结构）
    logit = (
        0.8 * X_data['tg'] + 
        0.7 * X_data['tc'] + 
        0.6 * X_data['ldl_c'] + 
        (-0.5) * X_data['hdl_c'] + 
        0.4 * X_data['urea_acid'] + 
        0.3 * X_data['tan_score'] + 
        0.2 * X_data['bmi']
    )
    prob = 1 / (1 + np.exp(-logit))
    y_data = pd.Series(np.random.binomial(1, prob))
    
    print(f"模拟数据: {X_data.shape}, 阳性比例: {y_data.mean():.2%}")
    print()
    
    identifier = KeyRiskFactorIdentifier()
    result = identifier.fit_all(X_data, y_data)
    
    print("\n特征权重汇总:")
    summary = identifier.get_feature_weight_summary(result)
    print(summary[['feature', 'lasso_coef', 'xgb_importance', 'rf_importance']].head(10))
