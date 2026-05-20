import json
import os
import random
# 提示用户如果未安装jieba先进行安装 pip install jieba
try:
    import jieba
except ImportError:
    print("需要安装jieba来生成测试用的参考参考标准答案(ground truth)。请运行: pip install jieba")
    exit(1)

def extract_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # 简单按句号感叹号分句
    sentences = text.replace('\n', '').replace('　', '').split('。')
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) > 5:
            s += '。' # 加回句号
            result.append(s)
    return result

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    input_dir = os.path.join(root_dir, 'input')
    output_file = os.path.join(root_dir, 'data', 'custom_corpus.json')
    
    files = ['WanderEarth.txt', 'Wekipedia.txt']
    
    custom_dataset = []
    
    # 我们设置随机种子以保证复现
    random.seed(42)
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        sentences = extract_sentences(file_path)
        
        # 针对每个类型（文件）抽取10句话
        sampled_sentences = random.sample(sentences, min(25, len(sentences)))
        
        for s in sampled_sentences:
            # 借助分词工具生成伪Golden Truth，方便评估。
            gt_words = list(jieba.cut(s))
            custom_dataset.append({
                "type": "Sci-Fi" if "Wander" in filename else "Wikipedia",
                "raw_sentence": s,
                "gt_words": gt_words
            })
            
    # 保存该自定义测试集
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(custom_dataset, f, ensure_ascii=False, indent=4)
        
    print(f"成功抽取自备网络语料并生成评测数据，共 {len(custom_dataset)} 条，已保存至 {output_file}")

if __name__ == "__main__":
    main()
