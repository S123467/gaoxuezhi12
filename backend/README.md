# 高血脂风险分层预警与干预推荐系统 - 后端

## 项目介绍

本项目基于数模论文《基于中西医融合信息的高血脂风险识别、分层预警与干预优化研究》（队伍编号: CMC2604725），完整复现论文三大核心算法：

1. **高血脂关键风险因素识别** - Lasso-Logistic 回归 + XGBoost-SHAP + 随机森林交叉验证
2. **三级风险分层预警** - CART决策树 + 中西医融合三级标签，论文验证准确率 **95.00%**
3. **个体化干预方案优化** - 画像分型（代谢异常/肥胖并发/功能受限）+ 约束组合优化，6个月成本 ≤ 2000元

## 目录结构

```
backend/
├── app.py                    # Flask API入口
├── requirements.txt          # Python依赖
├── config/
│   ├── risk_thresholds.json      # 论文提取的风险阈值
│   └── intervention_costs.json   # 干预成本效果参数
├── models/
│   ├── feature_identification.py  # 模块一：关键风险因素识别
│   ├── risk_classification.py     # 模块二：三级风险分层预警
│   └── intervention_optimization.py  # 模块三：个体化干预优化
└── utils/
    ├── data_loader.py          # 数据加载工具
    └── __init__.py
```

## 安装与运行

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

服务启动后，API将在 `http://localhost:5000` 监听。

## API接口说明

### 1. 完整预测 `POST /api/predict`

**输入示例:**
```json
{
  "age": 55,
  "gender": 1,
  "height": 170,
  "weight": 75,
  "tan_score": 45,
  "tg": 1.5,
  "tc": 5.2,
  "hdl_c": 1.2,
  "ldl_c": 3.4,
  "urea_acid": 360,
  "adl_score": 55,
  "budget": 2000
}
```

**输出:**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "input": { /* 输入摘要 */ },
    "risk": {
      "risk_level_code": 1,
      "risk_level": "medium",
      "risk_level_name": "中风险",
      "risk_factors": [ /* 异常因素列表 */ ],
      "description": "部分指标异常，需要关注..."
    },
    "patient_type": {
      "patient_type": "obesity",
      "type_name": "肥胖并发型",
      "recommended_strategy": "调理 + 运动联合干预"
    },
    "intervention": {
      "optimal_plan": { /* 最优方案 */ },
      "recommendation": "### 推荐方案...",
      "max_budget": 2000
    },
    "alternatives": [ /* 备选方案列表 */ ],
    "paper_accuracy": 0.95
  }
}
```

### 2. 关键因素识别 `POST /api/identify`

传入标注数据，执行完整的三步特征识别流程：
- Lasso-Logistic 特征筛选
- XGBoost-SHAP 非线性重要性分析
- 随机森林交叉验证

### 3. 获取阈值 `GET /api/thresholds`

获取论文提取的风险阈值配置。

## 算法对照论文

| 论文问题 | 算法 | 对应代码文件 |
|----------|------|--------------|
| 问题一 | Lasso-Logistic + XGBoost-SHAP + 随机森林交叉验证 | `models/feature_identification.py` |
| 问题二 | CART决策树三级风险分层 | `models/risk_classification.py` |
| 问题三 | 画像分型 + 约束组合优化 | `models/intervention_optimization.py` |

## 关键阈值（来自论文）

| 指标 | 高风险界值 | 单位 |
|------|-----------|------|
| 痰湿积分 | ≈ 60 | 分 |
| TG (甘油三酯) | ≈ 1.7 | mmol/L |
| TC (总胆固醇) | ≈ 6.2 | mmol/L |
| 活动能力 | ≈ 40 | 分 |

## 患者分型策略（来自论文）

| 患者类型 | 特征 | 推荐策略 |
|----------|------|----------|
| 代谢异常主导型 | 血脂异常为主，体质和活动能力尚可 | 优先强化调理 |
| 肥胖并发型 | BMI超标 + 痰湿质 + 代谢异常 | 调理 + 运动联合干预 |
| 功能受限型 | 活动能力下降 + 高龄 + 多种并发症 | 低强度稳步改善方案 |

## 依赖说明

- `numpy pandas scipy scikit-learn` - 基础数据科学
- `xgboost shap` - 机器学习和特征解释
- `flask flask-cors` - Web框架和跨域支持

## 准确性说明

本后端代码100%按照论文算法实现，不修改论文逻辑，阈值参数完全来自论文结论。CART决策树论文验证准确率为 **95.00%**。
