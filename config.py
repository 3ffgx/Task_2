from pathlib import Path


# 根目录：以Task_2 文件夹为基准。
BASE_DIR = Path(__file__).resolve().parent


DATA_PATH = BASE_DIR / "data" / "data_task2.csv" # 数据文件位置
RESULT_DIR = BASE_DIR / "results" #输出目录


READ_ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"] # 防止因编码方式导致乱码


RANDOM_STATE = 42 # 随机种子，无实际意义，行业默认
BOOTSTRAP_ITER = 500 # 置信区间计算重复500次
ROBUSTNESS_SEEDS = [42, 123, 456, 789, 1011] # 稳健性检验用的多个随机种子
VIF_THRESHOLD = 10.0 # VIF阈值



TARGET_COL = "总水资源生态足迹" # 设定预测目标为：总水资源生态足迹

# 泄露控制：防止强线性相关特征进入模型
LEAKAGE_RULES = {
    "总水资源生态足迹":
    [
        "人均水资源生态足迹",
        "压力指数",
        "水资源生态盈余赤字",
        "生活用水",
        "生态用水",
    ],
}
