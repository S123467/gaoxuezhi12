"""
高血脂风险分层预警与干预推荐系统 - Windows 桌面应用 GUI
基于 PyQt5 构建，可打包为独立 exe 运行

Author: 基于论文CMC2604725研究成果
Version: 1.0
"""
import sys
import json
from typing import Dict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QRadioButton,
    QButtonGroup, QPushButton, QTabWidget, QScrollArea,
    QGroupBox, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入核心算法模块
sys.path.append('..')
from backend.models.feature_identification import KeyRiskFactorIdentifier
from backend.models.risk_classification import ThreeLevelRiskClassifier
from backend.models.intervention_optimization import InterventionOptimizer


class PredictionWorker(QThread):
    """后台计算线程"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def run(self):
        # 计算风险
        self.progress.emit(10)
        classifier = ThreeLevelRiskClassifier()
        risk_result = classifier.predict_single(self.data['features'])
        self.progress.emit(50)

        # 患者分型
        self.progress.emit(70)
        optimizer = InterventionOptimizer()
        patient_type = optimizer.classify_patient(self.data['features'])
        self.progress.emit(85)

        # 干预优化
        intervention_result = optimizer.optimize(
            patient_type['patient_type'],
            max_budget=self.data['budget']
        )
        alternatives = optimizer.get_top_n_plans(
            patient_type['patient_type'],
            max_budget=self.data['budget'],
            n=3
        )
        self.progress.emit(100)

        result = {
            'risk': risk_result,
            'patient_type': patient_type,
            'intervention': intervention_result,
            'alternatives': alternatives
        }
        self.finished.emit(result)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("高血脂风险分层预警与干预推荐系统 V1.0")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        self.init_ui()
        self.result = None

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        # 标签
        title = QLabel("<h2>高血脂风险分层预警与干预推荐系统 V1.0</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("基于中西医融合信息 · 论文编号: CMC2604725")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        # 选项卡
        tabs = QTabWidget()

        # 选项卡1: 信息录入
        tab_input = QWidget()
        layout_input = QVBoxLayout()

        # 基本信息
        group_basic = QGroupBox("基本信息")
        layout_basic = QVBoxLayout()

        # 年龄
        layout_age = QHBoxLayout()
        layout_age.addWidget(QLabel("年龄 (岁):"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(18, 120)
        self.age_spin.setValue(55)
        layout_age.addWidget(self.age_spin)
        layout_basic.addLayout(layout_age)

        # 性别
        layout_gender = QHBoxLayout()
        layout_gender.addWidget(QLabel("性别:"))
        self.gender_group = QButtonGroup()
        self.radio_male = QRadioButton("男")
        self.radio_female = QRadioButton("女")
        self.radio_male.setChecked(True)
        self.gender_group.addButton(self.radio_male)
        self.gender_group.addButton(self.radio_female)
        layout_gender.addWidget(self.radio_male)
        layout_gender.addWidget(self.radio_female)
        layout_basic.addLayout(layout_gender)

        # 身高体重
        layout_hw = QHBoxLayout()
        layout_hw.addWidget(QLabel("身高 (cm):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(100, 220)
        self.height_spin.setValue(170)
        layout_hw.addWidget(self.height_spin)
        layout_hw.addWidget(QLabel("体重 (kg):"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(30, 200)
        self.weight_spin.setValue(75)
        layout_hw.addWidget(self.weight_spin)
        layout_basic.addLayout(layout_hw)

        group_basic.setLayout(layout_basic)
        layout_input.addWidget(group_basic)

        # 生化指标
        group_biochem = QGroupBox("生化指标 (最近一次体检)")
        layout_biochem = QVBoxLayout()

        layout_tg = QHBoxLayout()
        layout_tg.addWidget(QLabel("甘油三酯 (TG, mmol/L):"))
        self.tg_spin = QDoubleSpinBox()
        self.tg_spin.setRange(0, 20)
        self.tg_spin.setValue(1.5)
        self.tg_spin.setSingleStep(0.01)
        layout_tg.addWidget(self.tg_spin)
        layout_biochem.addLayout(layout_tg)

        layout_tc = QHBoxLayout()
        layout_tc.addWidget(QLabel("总胆固醇 (TC, mmol/L):"))
        self.tc_spin = QDoubleSpinBox()
        self.tc_spin.setRange(0, 15)
        self.tc_spin.setValue(5.2)
        self.tc_spin.setSingleStep(0.01)
        layout_tc.addWidget(self.tc_spin)
        layout_biochem.addLayout(layout_tc)

        layout_hdl = QHBoxLayout()
        layout_hdl.addWidget(QLabel("HDL-C (mmol/L):"))
        self.hdl_spin = QDoubleSpinBox()
        self.hdl_spin.setRange(0, 5)
        self.hdl_spin.setValue(1.2)
        self.hdl_spin.setSingleStep(0.01)
        layout_hdl.addWidget(self.hdl_spin)
        layout_biochem.addLayout(layout_hdl)

        layout_ldl = QHBoxLayout()
        layout_ldl.addWidget(QLabel("LDL-C (mmol/L):"))
        self.ldl_spin = QDoubleSpinBox()
        self.ldl_spin.setRange(0, 10)
        self.ldl_spin.setValue(3.4)
        self.ldl_spin.setSingleStep(0.01)
        layout_ldl.addWidget(self.ldl_spin)
        layout_biochem.addLayout(layout_ldl)

        layout_urea = QHBoxLayout()
        layout_urea.addWidget(QLabel("血尿酸 (μmol/L):"))
        self.urea_spin = QSpinBox()
        self.urea_spin.setRange(100, 800)
        self.urea_spin.setValue(360)
        layout_urea.addWidget(self.urea_spin)
        layout_biochem.addLayout(layout_urea)

        group_biochem.setLayout(layout_biochem)
        layout_input.addWidget(group_biochem)

        # 中医体质与活动能力
        group_cm = QGroupBox("中医体质与活动能力")
        layout_cm = QVBoxLayout()

        layout_tan = QHBoxLayout()
        layout_tan.addWidget(QLabel("痰湿体质积分 (0-100):"))
        self.tan_spin = QSpinBox()
        self.tan_spin.setRange(0, 100)
        self.tan_spin.setValue(45)
        layout_tan.addWidget(self.tan_spin)
        layout_cm.addLayout(layout_tan)

        layout_adl = QHBoxLayout()
        layout_adl.addWidget(QLabel("ADL活动能力评分 (0-100):"))
        self.adl_spin = QSpinBox()
        self.adl_spin.setRange(0, 100)
        self.adl_spin.setValue(55)
        layout_adl.addWidget(self.adl_spin)
        layout_cm.addLayout(layout_adl)

        group_cm.setLayout(layout_cm)
        layout_input.addWidget(group_cm)

        # 预算
        group_budget = QGroupBox("干预预算")
        layout_budget = QHBoxLayout()
        layout_budget.addWidget(QLabel("6个月预算上限 (元):"))
        self.budget_spin = QSpinBox()
        self.budget_spin.setRange(0, 5000)
        self.budget_spin.setSingleStep(100)
        self.budget_spin.setValue(2000)
        layout_budget.addWidget(self.budget_spin)
        group_budget.setLayout(layout_budget)
        layout_input.addWidget(group_budget)

        # 计算按钮
        self.calc_btn = QPushButton("开始计算风险")
        self.calc_btn.clicked.connect(self.calculate_risk)
        layout_input.addWidget(self.calc_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout_input.addWidget(self.progress_bar)

        tab_input.setLayout(layout_input)
        tabs.addTab(tab_input, "📝 信息录入")

        # 选项卡2: 风险结果
        tab_result = QScrollArea()
        tab_result.setWidgetResizable(True)
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout()

        # 风险等级
        self.risk_group = QGroupBox("风险分级结果")
        self.risk_layout = QVBoxLayout()
        self.risk_label = QLabel("")
        self.risk_label.setStyleSheet("font-size: 18px; font-weight: bold; text-align: center; padding: 10px;")
        self.risk_label.setAlignment(Qt.AlignCenter)
        self.risk_layout.addWidget(self.risk_label)
        self.risk_desc = QLabel("")
        self.risk_desc.setAlignment(Qt.AlignCenter)
        self.risk_layout.addWidget(self.risk_desc)
        self.risk_factors_box = QVBoxLayout()
        self.risk_layout.addWidget(QLabel("<strong>异常风险因素:</strong>"))
        self.risk_layout.addLayout(self.risk_factors_box)
        self.risk_group.setLayout(self.risk_layout)
        self.result_layout.addWidget(self.risk_group)

        # 患者分型
        self.type_group = QGroupBox("患者画像分型")
        self.type_layout = QVBoxLayout()
        self.type_name_label = QLabel("")
        self.type_name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.type_layout.addWidget(self.type_name_label)
        self.type_desc_label = QLabel("")
        self.type_layout.addWidget(self.type_desc_label)
        self.type_strategy_label = QLabel("")
        self.type_layout.addWidget(self.type_strategy_label)
        self.paper_conclusion_label = QLabel("")
        self.paper_conclusion_label.setStyleSheet("color: #666; font-style: italic;")
        self.type_layout.addWidget(self.paper_conclusion_label)
        self.type_group.setLayout(self.type_layout)
        self.result_layout.addWidget(self.type_group)

        self.result_widget.setLayout(self.result_layout)
        tab_result.setWidget(self.result_widget)
        tabs.addTab(tab_result, "📊 风险结果")

        # 选项卡3: 干预推荐
        tab_intervention = QScrollArea()
        tab_intervention.setWidgetResizable(True)
        self.interv_widget = QWidget()
        self.interv_layout = QVBoxLayout()

        # 最优推荐
        self.optimal_group = QGroupBox("📌 推荐最优方案 (预算约束下效果最大化)")
        self.optimal_layout = QVBoxLayout()
        self.optimal_text = QLabel("")
        self.optimal_text.setTextFormat(Qt.RichText)
        self.optimal_layout.addWidget(self.optimal_text)
        self.optimal_group.setLayout(self.optimal_layout)
        self.interv_layout.addWidget(self.optimal_group)

        # 备选方案
        self.alt_group = QGroupBox("其他备选方案")
        self.alt_layout = QVBoxLayout()
        self.alt_group.setLayout(self.alt_layout)
        self.interv_layout.addWidget(self.alt_group)

        # 输入摘要
        self.summary_group = QGroupBox("输入信息摘要")
        self.summary_layout = QVBoxLayout()
        self.summary_text = QLabel("")
        self.summary_layout.addWidget(self.summary_text)
        self.summary_group.setLayout(self.summary_layout)
        self.interv_layout.addWidget(self.summary_group)

        # 提示信息
        note = QLabel(
            "<br><div style='color: #666; padding: 10px;'>"
            "<strong>提示:</strong> 本系统基于学术研究成果开发，仅供研究参考，不构成临床诊断建议。"
            "如有健康问题，请咨询专业医疗机构和医师。"
            "</div>"
        )
        self.interv_layout.addWidget(note)

        self.interv_widget.setLayout(self.interv_layout)
        tab_intervention.setWidget(self.interv_widget)
        tabs.addTab(tab_intervention, "💡 干预推荐")

        # 选项卡4: 关于
        tab_about = QWidget()
        about_layout = QVBoxLayout()
        about_text = QLabel("""
        <h3>高血脂风险分层预警与干预推荐系统 V1.0</h3>
        <p><b>基于论文:</b> 基于中西医融合信息的高血脂风险识别、分层预警与干预优化研究<br>
        <b>队伍编号:</b> CMC2604725</p>

        <p><b>核心算法:</b></p>
        <ul>
        <li>关键风险因素识别: Lasso-Logistic + XGBoost-SHAP + 随机森林交叉验证</li>
        <li>三级风险分层预警: CART决策树，论文验证准确率 95.00%</li>
        <li>个体化干预优化: 画像分型 + 约束组合优化，6个月成本 ≤ 2000元</li>
        </ul>

        <p><b>患者分型策略:</b></p>
        <ul>
        <li><b>代谢异常主导型</b> → 优先强化调理</li>
        <li><b>肥胖并发型</b> → 调理 + 运动联合干预</li>
        <li><b>功能受限型</b> → 低强度稳步改善方案</li>
        </ul>
        """)
        about_text.setTextFormat(Qt.RichText)
        about_layout.addWidget(about_text)
        tab_about.setLayout(about_layout)
        tabs.addTab(tab_about, "ℹ️ 关于")

        layout.addWidget(tabs)
        self.tabs = tabs
        central.setLayout(layout)
        self.setCentralWidget(central)

    def get_input_data(self) -> Dict:
        """收集输入数据"""
        gender = 1 if self.radio_male.isChecked() else 0
        return {
            'age': self.age_spin.value(),
            'gender': gender,
            'height': self.height_spin.value(),
            'weight': self.weight_spin.value(),
            'tan_score': self.tan_spin.value(),
            'tg': self.tg_spin.value(),
            'tc': self.tc_spin.value(),
            'hdl_c': self.hdl_spin.value(),
            'ldl_c': self.ldl_spin.value(),
            'urea_acid': self.urea_spin.value(),
            'adl_score': self.adl_spin.value(),
            'budget': self.budget_spin.value()
        }

    def calculate_risk(self):
        """开始计算"""
        data = self.get_input_data()

        # 计算BMI
        height_m = data['height'] / 100
        bmi = data['weight'] / (height_m ** 2)
        data['bmi'] = bmi
        data['features'] = {
            'age': data['age'],
            'gender': data['gender'],
            'bmi': bmi,
            'tan_score_raw': data['tan_score'],
            'tg': data['tg'],
            'tc': data['tc'],
            'hdl_c': data['hdl_c'],
            'ldl_c': data['ldl_c'],
            'urea_acid': data['urea_acid'],
            'activity_score': data['adl_score']
        }

        self.calc_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = PredictionWorker(data)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, value):
        self.progress_bar.setValue(value)

    def on_prediction_finished(self, result):
        self.result = result
        self.calc_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 显示风险结果
        risk = result['risk']
        risk_color = risk['risk_color']
        self.risk_label.setText(risk['risk_level_name'])
        self.risk_label.setStyleSheet(
            f"background-color: {risk_color}; color: white; font-size: 18px; "
            f"font-weight: bold; text-align: center; padding: 10px; border-radius: 5px;"
        )
        self.risk_desc.setText(risk['description'])

        # 清空旧的风险因素
        for i in reversed(range(self.risk_factors_box.count())):
            self.risk_factors_box.itemAt(i).widget().setParent(None)

        # 添加风险因素
        if risk['risk_factors']:
            for factor in risk['risk_factors']:
                label = QLabel(f"⚠️ {factor}")
                label.setStyleSheet("background-color: #fff3cd; padding: 8px; border-radius: 3px;")
                self.risk_factors_box.addWidget(label)
        else:
            label = QLabel("✅ 未发现明显异常风险因素")
            label.setStyleSheet("color: #28a745; padding: 8px;")
            self.risk_factors.addWidget(label)

        # 显示患者分型
        pt = result['patient_type']
        self.type_name_label.setText(f"🏷️ {pt['type_name']}")
        self.type_desc_label.setText(pt['description'])
        self.type_strategy_label.setText(f"<strong>推荐策略:</strong> {pt['recommended_strategy']}")
        self.paper_conclusion_label.setText(pt['paper_conclusion'])

        # 显示干预推荐
        interv = result['intervention']
        optimal = interv['optimal_plan']
        if optimal:
            rec_text = f"""
            <table cellpadding="5" style="width: 100%;">
                <tr><td><strong>调理方案:</strong></td><td>{optimal['treatment']['name']}</td></tr>
                <tr><td><strong>调理内容:</strong></td><td>{optimal['treatment']['description']}</td></tr>
                <tr><td><strong>月费用:</strong></td><td>{optimal['treatment']['cost_per_month']} 元</td></tr>
            """
            if optimal['exercise']:
                rec_text += f"""
                <tr><td><strong>运动方案:</strong></td><td>{optimal['intensity_name']}, {optimal['frequency_name']}</td></tr>
                <tr><td><strong>运动月费用:</strong></td><td>{optimal['exercise']['cost_per_month']} 元</td></tr>
                """
            rec_text += f"""
                <tr><td><strong>6个月总成本:</strong></td><td><strong>{optimal['total_cost_6months']:.0f} 元</strong></td></tr>
                <tr><td><strong>预期总效果:</strong></td><td>{optimal['total_effect_6months']:.1f} 单位 (痰湿积分改善预期)</td></tr>
            </table>
            """
            self.optimal_text.setText(rec_text)
        else:
            self.optimal_text.setText("无法找到满足约束的方案")

        # 清空备选方案
        for i in reversed(range(self.alt_layout.count())):
            self.alt_layout.itemAt(i).widget().setParent(None)

        alternatives = result['alternatives']
        if len(alternatives) > 1:
            for idx, plan in enumerate(alternatives[1:]):
                alt_html = f"""
                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 8px;">
                <strong>备选 {idx+1}:</strong> {plan['treatment_name']}<br>
                <strong>运动:</strong> {plan['exercise_fullname']}<br>
                6个月成本: <strong>{plan['total_cost_6months']:.0f}</strong> 元 |
                预期效果: <strong>{plan['total_effect_6months']:.1f}</strong> 单位
                </div>
                """
                label = QLabel(alt_html)
                label.setTextFormat(Qt.RichText)
                self.alt_layout.addWidget(label)
        else:
            label = QLabel("没有其他备选方案")
            self.alt_layout.addWidget(label)

        # 摘要
        summary_html = f"""
        <table style="width: 100%;">
        <tr><td><strong>年龄</strong></td><td>{data['age']} 岁</td><td><strong>BMI</strong></td><td>{bmi:.1f} kg/m²</td></tr>
        <tr><td><strong>痰湿积分</strong></td><td>{data['tan_score']} 分</td><td><strong>活动能力</strong></td><td>{data['adl_score']} 分</td></tr>
        <tr><td><strong>TG</strong></td><td>{data['tg']:.1f} mmol/L</td><td><strong>TC</strong></td><td>{data['tc']:.1f} mmol/L</td></tr>
        <tr><td><strong>预算上限</strong></td><td colspan="3">{data['budget']} 元/6个月</td></tr>
        </table>
        """
        self.summary_text.setText(summary_html)
        self.summary_text.setTextFormat(Qt.RichText)

        # 切换到结果选项卡
        self.tabs.setCurrentIndex(1)

        QMessageBox.information(self, "计算完成", "风险计算完成，请查看「风险结果」和「干预推荐」选项卡。")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
