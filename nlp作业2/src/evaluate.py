import torch
import torch.nn.functional as F
import random

def get_top_similar(model_file, target_word, topn=10, f_out=None):
    data = torch.load(model_file)
    embeddings = data["embeddings"]
    word_to_idx = data["word_to_idx"]
    idx_to_word = data["idx_to_word"]

    if target_word not in word_to_idx:
        msg = f"词 {target_word} 不在词表中\n"
        if f_out: f_out.write(msg)
        else: print(msg, end="")
        return

    target_vec = embeddings[word_to_idx[target_word]].unsqueeze(0)
    sims = F.cosine_similarity(target_vec, embeddings)

    top_idx = torch.topk(sims, topn+1).indices.tolist()
    res = f"\n【{target_word}】的相似词：\n"
    for idx in top_idx:
        if idx == word_to_idx[target_word]:
            continue
        res += f"  {idx_to_word[idx]}  (相似度={sims[idx]:.4f})\n"
    
    if f_out:
        f_out.write(res)
    else:
        print(res, end="")

def random_test(model_file, n=20, f_out=None):
    data = torch.load(model_file)
    idx_to_word = data["idx_to_word"]
    vocab = list(idx_to_word.values())
    words = random.sample(vocab, n)
    for w in words:
        get_top_similar(model_file, w, 10, f_out=f_out)

if __name__ == "__main__":
    models = ["FNN", "RNN", "LSTM"]
    
    for model_name in models:
        model_file = f"{model_name}_embeddings.pt"
        out_file = f"{model_name}_evaluate.txt"
        
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"=== {model_name} 模型测试结果 ===\n")
                get_top_similar(model_file, "中国", 10, f_out=f)
                
                f.write("\n--- 随机测试 ---\n")
                random_test(model_file, 20, f_out=f)
            print(f"成功导出 {model_name} 的结果到 {out_file}")
        except FileNotFoundError:
            print(f"跳过 {model_name}: 找不到模型文件 {model_file} (请先训练该模型)")
        except Exception as e:
            print(f"无法处理 {model_name}，错误: {e}")