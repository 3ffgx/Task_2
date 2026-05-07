# Task_2

当前版本的核心设定如下：

- 预测目标固定为 `总水资源生态足迹`
- 采用轻度特征屏蔽方案，默认剔除 `生活用水` 和 `生态用水`
- 将原始大脚本拆分为配置、预处理、建模、绘图四个模块，便于复现和维护

## 目录结构

```text
Task_2/
├─ config.py
├─ preprocess.py
├─ modeling.py
├─ plotting.py
├─ run_model.py
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ data_task2.csv
└─ results/
```

各文件作用如下：

- `config.py`：统一管理数据路径、结果路径、目标变量、特征屏蔽规则和随机种子
- `preprocess.py`：负责数据读取、候选特征构建、相关性筛选和 VIF 筛选
- `modeling.py`：负责 LOOCV 建模、指标计算、稳健性检验、未来三年预测和结果导出
- `plotting.py`：负责热力图、VIF 图、模型对比图、残差图、特征重要性图等绘制
- `run_model.py`：主入口脚本，串联完整流程

## 数据准备

请将建模数据放在：

```text
Task_2/data/data_task2.csv
```

## 环境依赖

建议使用：

- `Python 3.10+`

安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方法

进入 `Task_2` 目录后执行：

```bash
python run_model.py
```

运行完成后，结果会自动输出到：

```text
Task_2/results/
```

## 默认建模设置

当前默认配置位于 [config.py](./config.py)：

- 目标变量：`总水资源生态足迹`
- 泄漏控制：剔除 `可持续利用指数`
- 额外屏蔽特征：`生活用水`、`生态用水`
- VIF 阈值：`10.0`

## 典型输出结果

运行后通常会生成以下结果文件：

- `00_预处理摘要.xlsx`
- `00_VIF检验结果.xlsx`
- `00_相关性矩阵.xlsx`
- `01_模型评估结果.xlsx`
- `02_预测结果对照表.xlsx`
- `03_ET最优参数.xlsx`
- `03_RF最优参数.xlsx`
- `04_ET特征重要度.xlsx`
- `04_RF特征重要度.xlsx`
- `05_稳健性检验结果.xlsx`
- `06_ET_SHAP特征重要度.xlsx`
- `07_生态用水补充敏感性分析.xlsx`
- `08_future_3yr_forecast.xlsx`
- 各类 PNG 图件

## 说明

- 如果想快速理解主流程，建议先看 [run_model.py](./run_model.py) 和 [config.py](./config.py)。
