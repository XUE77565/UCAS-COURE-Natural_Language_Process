# NLP 作业 3 - 大模型分词测试指南

## 目录结构
```text
nlp作业3/
├── input/
│   └── ChineseCorpus199801.txt   # 原训练语料（你已经放进来了）
├── data/
│   └── sampled_50.json           # 执行脚步后自动提取的50句测试集（包含原始句和切分标准答案）
├── scripts/
│   ├── data_process.py           # 预处理与抽样脚本
│   ├── evaluate.py               # Precision, Recall, F1 计算指标脚本
│   └── task1_llm_pku.py          # 调用大模型执行任务1的核心脚本
└── README.md
```

## 1.安装Openai
```bash
pip install openai
```

## 2.生成测试数据集
```bash
python scripts/data_process.py
```
*执行后，你会在 `data/` 目录下看到新生成的 `sampled_50.json`

## 3.经典语料分词
打开 `scripts/task1_llm_pku.py`，在代码头部填入你自己申请的 `API_KEY`：
*   把 `API_KEY`、`BASE_URL` 和 `MODEL_NAME` 替换为你所选平台的对应值。
*   运行：
```bash
python scripts/task1_llm_pku.py
```