import json
import os
import time
from evaluate import evaluate_dataset

from openai import OpenAI

# ----------------- 配置区 -----------------
API_KEY = "sk-9bc1c11d39214f268fa4c628361528ee" 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
# ----------------------------------------

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def segment_sentence_with_llm(sentence):
    prompt = f"你是一流的中文语言学专家。请对下面的句子进行中文分词，词与词之间请用空格隔开，保留所有标点符号。不要输出任何解释性的废话，，只输出分词后的纯文本结果。\n\n句子：{sentence}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.01,
        )
        result = response.choices[0].message.content.strip()
        return result.split()
    except Exception as e:
        print(f"调用 API 失败: {e}")
        return []

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(root_dir, 'data', 'custom_corpus.json')
    if not os.path.exists(data_file):
        print(f"找不到数据文件 {data_file}，请先运行 src/data_process_custom.py")
        return
        
    with open(data_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # 按语料类型分离
    results_by_type = {}
    
    print(f"开始使用 {MODEL_NAME} 对网络不同类型语料进行分词...")
    for i, item in enumerate(dataset):
        text_type = item.get("type", "Unknown")
        sentence = item['raw_sentence']
        gt = item['gt_words']
        
        print(f"正在处理第 {i+1} 句 ({text_type}): {sentence[:20]}...")
        pred = segment_sentence_with_llm(sentence)
        time.sleep(0.5) 
        
        if text_type not in results_by_type:
            results_by_type[text_type] = {"preds": [], "gts": [], "details": []}
            
        results_by_type[text_type]["preds"].append(pred)
        results_by_type[text_type]["gts"].append(gt)
        results_by_type[text_type]["details"].append({
            "原句": sentence,
            "参考标准答案(Jieba)": " ".join(gt),
            f"{MODEL_NAME}预测": " ".join(pred)
        })
    
    # 分类计算并打印
    final_output = {"overall": {}, "by_type": {}}
    overall_preds = []
    overall_gts = []
    
    print("\n========== 任务二 评测结果 (不同文本类型) ==========")
    for text_type, data in results_by_type.items():
        avg_p, avg_r, avg_f1 = evaluate_dataset(data["preds"], data["gts"])
        print(f"[{text_type}] Precision: {avg_p:.4f}, Recall: {avg_r:.4f}, F1-score: {avg_f1:.4f}")
        
        final_output["by_type"][text_type] = {
            "metrics": {
                "precision": round(avg_p, 4),
                "recall": round(avg_r, 4),
                "f1_score": round(avg_f1, 4)
            },
            "details": data["details"]
        }
        
        overall_preds.extend(data["preds"])
        overall_gts.extend(data["gts"])
        
    # 计算总平均
    o_p, o_r, o_f1 = evaluate_dataset(overall_preds, overall_gts)
    print(f"[总体] Precision: {o_p:.4f}, Recall: {o_r:.4f}, F1-score: {o_f1:.4f}")
    
    final_output["overall"] = {
        "precision": round(o_p, 4),
        "recall": round(o_r, 4),
        "f1_score": round(o_f1, 4)
    }
    
    output_comparison_file = os.path.join(root_dir, 'data', f'task2_result_{MODEL_NAME}.json')
    with open(output_comparison_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 任务二 结果已保存至: {output_comparison_file}")

if __name__ == "__main__":
    if API_KEY != "你的API_KEY":
        main()
    else:
        print("请填入真实的 API_KEY 后运行。")
