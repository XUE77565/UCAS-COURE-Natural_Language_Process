import os
import torch

# 获取根目录路径 (假设 config.py 在 src 目录下)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ====== 基本配置 ======
CORPUS_PATH = os.path.join(BASE_DIR, "input", "ChineseCorpus199801.txt")
ENCODING = "gbk"                        # 如果乱码改为 "gbk"

# ====== 训练参数 ======
MIN_FREQ = 15
CONTEXT_SIZE = 5
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 512
EPOCHS = 50
LR = 0.001

# ====== 运行模式 ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
