import torch
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
from sklearn.decomposition import PCA
import torch.nn.functional as F

def load_embeddings(file):
    data = torch.load(file)
    return data["embeddings"], data["word_to_idx"], data["idx_to_word"]

def compare_similarity(word, files, topn=5, f_out=None):
    res = f"\n=== 词【{word}】在不同模型下的相似词对比 ===\n"
    for f in files:
        embeddings, word_to_idx, idx_to_word = load_embeddings(f)
        if word not in word_to_idx:
            res += f"{f}: 不包含该词\n"
            continue
        idx = word_to_idx[word]
        vec = embeddings[idx].unsqueeze(0)
        sims = F.cosine_similarity(vec, embeddings)
        top_idx = torch.topk(sims, topn+1).indices.tolist()
        res += f"\n模型 {f}:\n"
        for i in top_idx:
            if i == idx: 
                continue
            res += f"  {idx_to_word[i]}  (相似度={sims[i]:.4f})\n"
    if f_out:
        f_out.write(res)
    else:
        print(res, end="")

def plot_pca(words, files):
    plt.figure(figsize=(12,4))
    for i, f in enumerate(files):
        embeddings, word_to_idx, idx_to_word = load_embeddings(f)
        vectors = []
        labels = []
        for w in words:
            if w in word_to_idx:
                vectors.append(embeddings[word_to_idx[w]].numpy())
                labels.append(w)
        pca = PCA(n_components=2)
        vec2d = pca.fit_transform(vectors)

        plt.subplot(1, len(files), i+1)
        plt.title(f)
        for j, w in enumerate(labels):
            plt.scatter(vec2d[j,0], vec2d[j,1])
            plt.text(vec2d[j,0], vec2d[j,1], w)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    files = ["FNN_embeddings.pt", "RNN_embeddings.pt", "LSTM_embeddings.pt"]

    # 相似词对比
    with open("compare_results.txt", "w", encoding="utf-8") as f:
        compare_similarity("中国", files, topn=10, f_out=f)
    print("相似词对比结果已保存到 compare_results.txt 文件中。")

    # PCA 可视化对比（可自定义词）
    words = ["中国", "美国", "经济", "发展", "人民", "市场", "改革", "建设"]
    plot_pca(words, files)