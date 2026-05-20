import os
import json
import matplotlib.pyplot as plt

# 设置 matplotlib 中文字体以便在图中显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(root_dir, 'data')
    
    models_metrics = {}
    
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if not filename.endswith('.json'):
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if filename.startswith('comparison_result_'):
                model_name = filename.replace('comparison_result_', '').replace('.json', '')
                if "overall_metrics" in data:
                    label = f"{model_name}\n(无Prompt改良, PKU语料)"
                    models_metrics[label] = data["overall_metrics"]
                    
            elif filename.startswith('better_prompt_result_'):
                model_name = filename.replace('better_prompt_result_', '').replace('.json', '')
                if "overall_metrics" in data:
                    label = f"{model_name}\n(有Prompt改良, PKU语料)"
                    models_metrics[label] = data["overall_metrics"]
                    
            elif filename.startswith('task2_result_'):
                model_name = filename.replace('task2_result_', '').replace('.json', '')
                if "by_type" in data:
                    for corpus_type, type_data in data["by_type"].items():
                        label = f"{model_name}\n(无Prompt改良, {corpus_type}语料)"
                        if "metrics" in type_data:
                            models_metrics[label] = type_data["metrics"]
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            
    if not models_metrics:
        print("未在 data 目录下找到任何有效的评测结果 Json 文件。")
        return
        
    # 打印对比表格
    print("\n" + "="*80)
    print(" 任务三: 不同大语言模型分词性能对比分析")
    print("="*80)
    print(f"{'模型与条件':<45} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10}")
    print("-" * 80)
    for model, metrics in models_metrics.items():
        p = metrics.get('precision', 0)
        r = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        display_label = model.replace('\n', ' ')
        print(f"{display_label:<45} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
    print("="*80 + "\n")
    
    # 绘制对比柱状图
    models = list(models_metrics.keys())
    precisions = [models_metrics[m].get('precision', 0) for m in models]
    recalls = [models_metrics[m].get('recall', 0) for m in models]
    f1s = [models_metrics[m].get('f1_score', 0) for m in models]
    
    x = range(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rects1 = ax.bar([i - width for i in x], precisions, width, label='Precision')
    rects2 = ax.bar(x, recalls, width, label='Recall')
    rects3 = ax.bar([i + width for i in x], f1s, width, label='F1-score')
    
    ax.set_ylabel('Scores')
    ax.set_title('不同大模型在分词任务上的性能直方图对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right')
    ax.legend(loc='lower right')
    
    # 在柱状图上方标注具体数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plot_file = os.path.join(root_dir, 'report', 'fig', 'model_comparison.png')
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.savefig(plot_file, dpi=300)
    print(f"📊 模型性能对比图表已保存至: {plot_file}")
    print("您可以结合参数量（如DeepSeek-V3 vs DeepSeek-R1参数量、结构等），撰写实验报告中的对比分析了。")

if __name__ == "__main__":
    main()
