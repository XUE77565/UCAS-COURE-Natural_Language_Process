import math
from collections import Counter, defaultdict
import pandas as pd
from parse_1998 import iter_1998_sentences

def build_word_pmi(path_1998: str, k: int = 5, min_pair_freq: int = 5):
    unigram = Counter()
    pair = Counter()
    pair_by_pos = defaultdict(Counter)
    total_tokens = 0
    total_pairs = 0

    for sent in iter_1998_sentences(path_1998):
        toks = [(w,pos) for (w,pos) in sent if not (pos=="w")]  # 先去掉标点
        words = [w for w,_ in toks]
        poses = [pos for _,pos in toks]
        n = len(words)
        for w in words:
            unigram[w] += 1
            total_tokens += 1

        for i in range(n):
            for j in range(i+1, min(n, i+k+1)):
                w1, w2 = words[i], words[j]
                pos1, pos2 = poses[i], poses[j]
                pair[(w1,w2)] += 1
                total_pairs += 1
                # pos bucket（无序/有序你选；这里用有序）
                pair_by_pos[(pos1,pos2)][(w1,w2)] += 1

    def pmi(w1,w2,f12):
        px = unigram[w1] / total_tokens
        py = unigram[w2] / total_tokens
        pxy = f12 / total_pairs
        return math.log2(pxy/(px*py))

    # 输出总体 top
    rows = []
    for (w1,w2), f12 in pair.items():
        if f12 < min_pair_freq:
            continue
        rows.append((w1,w2,f12,pmi(w1,w2,f12)))
    rows.sort(key=lambda t:t[3], reverse=True)
    df_all = pd.DataFrame(rows[:500], columns=["w1","w2","f","pmi"])
    df_all.to_csv("out/word_pmi_top.csv", index=False, encoding="utf-8-sig")

    # 输出按词性组合 top（挑你最想展示的几个组合）
    for (pos1,pos2), c in pair_by_pos.items():
        rows2 = []
        for (w1,w2), f12 in c.items():
            if f12 < min_pair_freq:
                continue
            rows2.append((w1,w2,f12,pmi(w1,w2,f12)))
        rows2.sort(key=lambda t:t[3], reverse=True)
        if rows2:
            df = pd.DataFrame(rows2[:200], columns=["w1","w2","f","pmi"])
            df.to_csv(f"out/word_pmi_{pos1}_{pos2}_top.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    build_word_pmi("data/ChineseCorpus199801.txt", k=5, min_pair_freq=5)