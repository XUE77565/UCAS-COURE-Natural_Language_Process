import json
import os
import time
from evaluate import evaluate_dataset

# TODO: 请确保已安装并配置对应大模型库 (例如 pip install openai)
# 这里以兼容 OpenAI 接口的第三方模型 (如 DeepSeek, 通义千问, 智谱) 为例：
from openai import OpenAI

# ----------------- 配置区 -----------------
# 替换为你的 API_KEY 和 API_BASE_URL
API_KEY = "sk-9bc1c11d39214f268fa4c628361528ee" 
BASE_URL = "https://api.deepseek.com" # 示例: DeepSeek 的 Base URL
MODEL_NAME = "deepseek-chat" # 示例: 模型名称
# ----------------------------------------

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def segment_sentence_with_llm(sentence):
    """
    调用大语言模型进行分词
    """
    prompt = f"你是一流的中文语言学专家。请对下面的句子进行中文分词，词与词之间请用空格隔开，保留所有标点符号。不要输出任何解释性的废话，只输出分词后的纯文本结果。\n\n句子:{sentence}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.01, # 温度设低，要求输出稳定
        )
        # 获取大模型的输出
        result = response.choices[0].message.content.strip()
        # 大模型会用空格隔开词语，我们通过 split() 将其转为词列表
        predicted_words = result.split()
        return predicted_words
    except Exception as e:
        print(f"调用 API 失败: {e}")
        return []

def main():
    data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_50.json')
    if not os.path.exists(data_file):
        print(f"找不到数据文件 {data_file}，请先运行 data_process.py")
        return
        
    with open(data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    predictions = []
    ground_truths = []
    comparison_results = []
    
    print(f"开始使用 {MODEL_NAME} 对 {len(dataset)} 个句子进行分词...")
    for i, item in enumerate(dataset):
        sentence = item['raw_sentence']
        gt = item['gt_words']
        
        print(f"正在处理第 {i+1} 句: {sentence[:20]}...")
        pred = segment_sentence_with_llm(sentence)
        
        # 为了不触发 API 的频率限制，可以稍作停顿
        time.sleep(0.5) 
        
        predictions.append(pred)
        ground_truths.append(gt)
        
        # 记录对比数据
        comparison_results.append({
            "原句": sentence,
            "标准答案": " ".join(gt),
            "模型预测": " ".join(pred)
        })
    
    # 评测结果
    avg_p, avg_r, avg_f1 = evaluate_dataset(predictions, ground_truths)
    
    # 组合输出数据：包含整体指标和详细对比
    final_output = {
        "overall_metrics": {
            "precision": round(avg_p, 4),
            "recall": round(avg_r, 4),
            "f1_score": round(avg_f1, 4)
        },
        "details": comparison_results
    }
    
    # 将完整结果存入文件中
    output_comparison_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'better_prompt_result_{MODEL_NAME}.json')
    with open(output_comparison_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 详细分词对比结果及评测指标已保存至: {output_comparison_file}")
    
    print("\n========== 任务一 评测结果 (模型: {MODEL_NAME}) ==========")
    print(f"Precision (精确率): {avg_p:.4f}")
    print(f"Recall (召回率): {avg_r:.4f}")
    print(f"F1-score (F1值): {avg_f1:.4f}")


if __name__ == "__main__":
    # 如果还没有填好 API KEY，先不直接运行
    if API_KEY != "你的API_KEY":
        main()
    else:
        print("请在 scripts/task1_llm_pku.py 中填入真实的 API_KEY 后运行本测试。")