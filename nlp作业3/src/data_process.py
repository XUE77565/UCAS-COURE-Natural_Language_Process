import random
import json
import os

def process_corpus(input_path, output_path, sample_size=50):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(input_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()
    
    processed_data = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        tokens = line.split()
        if len(tokens) <= 1:
            continue
            
        # 跳过第一个 token (例如: 19980101-01-001-001/m)
        words_with_tags = tokens[1:]
        
        gt_words = []
        for token in words_with_tags:
            # 去除词性标注和方括号 (例如 "[中央/n" -> "中央", "电台/n]nt" -> "电台")
            # token.split('/')[0] 获取符号前面的词
            word = token.split('/')[0].replace('[', '')
            if word:  # 确保不是空字符串
                gt_words.append(word)
                
        raw_sentence = "".join(gt_words)
        
        # 过滤掉太短的句子，保证测试质量
        if len(raw_sentence) > 5:
            processed_data.append({
                "raw_sentence": raw_sentence,
                "gt_words": gt_words
            })
            
    # 随机抽取 50 个句子作为评测数据
    # 注释掉 random.seed(42) 或让它每次运行前使用不同的种子
    # random.seed() 不传入参数默认使用当前系统时间作为种子，每次抽取的都不一样
    random.seed(18)
    sampled_data = random.sample(processed_data, sample_size)
    
    # 确保存储目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, item in enumerate(sampled_data):
            f.write("    {\n")
            f.write(f'        "raw_sentence": {json.dumps(item["raw_sentence"], ensure_ascii=False)},\n')
            f.write(f'        "gt_words": {json.dumps(item["gt_words"], ensure_ascii=False)}\n')
            if i < len(sampled_data) - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")
        f.write("]\n")
        
    print(f"成功随机抽取 {sample_size} 个句子，已保存至 {output_path}")

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input', 'ChineseCorpus199801.txt')
    output_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampled_50.json')
    process_corpus(input_file, output_file)
