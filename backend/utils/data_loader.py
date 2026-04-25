"""
数据加载工具
用于加载训练数据和预处理
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def load_csv_data(
    file_path: str,
    label_column: str = 'label',
    separate_train_test: bool = False,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series] | Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    从CSV加载数据
    
    参数:
        file_path: CSV文件路径
        label_column: 标签列名
        separate_train_test: 是否拆分训练测试
        test_size: 测试集比例
        random_state: 随机种子
        
    返回:
        (X_train, y_train, X_test, y_test) 或者 (X, y)
    """
    df = pd.read_csv(file_path)
    
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    if not separate_train_test:
        return X, y
    
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, y_train, X_test, y_test


def check_data_quality(X: pd.DataFrame) -> dict:
    """
    检查数据质量
    
    参数:
        X: 特征矩阵
        
    返回:
        质量检查结果字典
    """
    result = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'missing_values': X.isnull().sum().to_dict(),
        'total_missing': X.isnull().sum().sum(),
        'constant_features': [],
        'highly_correlated': []
    }
    
    # 检查常数特征
    for col in X.columns:
        if X[col].nunique() <= 1:
            result['constant_features'].append(col)
    
    # 检查高度相关
    corr = X.corr().abs()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > 0.8:
                result['highly_correlated'].append((
                    corr.columns[i],
                    corr.columns[j],
                    corr.iloc[i, j]
                ))
    
    return result


def preprocess_features(
    X: pd.DataFrame,
    winsorize_cols: Optional[list] = None
) -> pd.DataFrame:
    """
    特征预处理，实现论文中的方法
    
    参数:
        X: 特征矩阵
        winsorize_cols: 需要缩尾处理的列
        
    返回:
        预处理后的特征矩阵
    """
    from scipy.stats.mstats import winsorize
    
    X_processed = X.copy()
    
    if winsorize_cols is None:
        # 默认对所有连续列进行缩尾处理
        winsorize_cols = X_processed.columns.tolist()
    
    for col in winsorize_cols:
        if col in X_processed.columns:
            X_processed[col] = winsorize(X_processed[col].values, limits=(0.01, 0.01))
    
    return X_processed
