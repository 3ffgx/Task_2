from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# 建议将建模数据放到 Task_2/data/data_task2.csv
DATA_PATH = BASE_DIR / "data" / "data_task2.csv"
RESULT_DIR = BASE_DIR / "results"

READ_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]

RANDOM_STATE = 42
BOOTSTRAP_ITER = 500
ROBUSTNESS_SEEDS = [42, 123, 456, 789, 1011]
VIF_THRESHOLD = 10.0

# 当前结构化版本固定预测目标为“总水资源生态足迹”
TARGET_COL = "总水资源生态足迹"

LEAKAGE_RULES = {
    "总水资源生态足迹": ["可持续利用指数"],
}

OPTIONAL_REDUNDANT_FEATURES = [
    "生活用水",
    "生态用水",
]
