import math
from collections import Counter, defaultdict
import pandas as pd
from parse_1998 import iter_1998_sentences

def build_char_stats(path_1998: str, max_d: int = 5):
    char_u = Counter()
    char_pair = {d: Counter() for d in range(1, max_d + 1)}
    adj = Counter()       # f_adj(x,y) = d=1
    inword = Counter()    # f_inword(x,y)

    total_chars = 0
    total_pairs_d = Counter()  # 记录每个 d 的总 pair 数

    for sent in iter_1998_sentences(path_1998):
        char_seq = []
        for w, pos in sent:
            # 1) 过滤标点符号（逗号、括号、句号等都是 /w）
            if pos == "w":
                continue

            # 2) 成词概率：只统计“词内部”的相邻字对
            if len(w) >= 2:
                for i in range(len(w) - 1):
                    inword[(w[i], w[i + 1])] += 1

            # 3) 字序列：把词拆成字
            char_seq.extend(list(w))

        # unigram
        for ch in char_seq:
            char_u[ch] += 1
        total_chars += len(char_seq)

        # pairs by distance
        n = len(char_seq)
        for d in range(1, max_d + 1):
            for i in range(n - d):
                x, y = char_seq[i], char_seq[i + d]
                char_pair[d][(x, y)] += 1
                total_pairs_d[d] += 1

    adj = char_pair[1]
    return char_u, char_pair, adj, inword, total_chars, total_pairs_d

def pmi_from_counts(char_u, pair_c, total_chars, total_pairs, min_pair_freq=5, logbase=2):
    log = math.log2 if logbase == 2 else math.log
    rows = []
    for (x, y), fxy in pair_c.items():
        if fxy < min_pair_freq:
            continue
        px = char_u[x] / total_chars
        py = char_u[y] / total_chars
        pxy = fxy / total_pairs
        score = log(pxy / (px * py))
        rows.append((x, y, fxy, score))
    rows.sort(key=lambda t: t[3], reverse=True)
    return rows

def sameword_prob(adj, inword, min_adj_freq=5):
    rows = []
    for (x, y), f_adj in adj.items():
        if f_adj < min_adj_freq:
            continue
        f_in = inword.get((x, y), 0)
        prob = f_in / f_adj
        rows.append((x, y, f_adj, f_in, prob))
    rows.sort(key=lambda t: (t[4], t[2]), reverse=True)
    return rows

if __name__ == "__main__":
    path = "data/ChineseCorpus199801.txt"
    char_u, char_pair, adj, inword, total_chars, total_pairs_d = build_char_stats(path, max_d=5)

    # PMI for d=1
    pmi_rows = pmi_from_counts(char_u, adj, total_chars, total_pairs_d[1], min_pair_freq=5, logbase=2)
    df_pmi = pd.DataFrame(pmi_rows[:500], columns=["x", "y", "f_xy", "pmi"])
    df_pmi.to_csv("out/char_pmi_d1_top.csv", index=False, encoding="utf-8-sig")

    # co-occur prob for d=1 top
    rows = [ (x, y, f, f/total_pairs_d[1]) for (x,y), f in adj.most_common(500) ]
    pd.DataFrame(rows, columns=["x","y","f_xy","p_d1"]).to_csv("out/char_cooccur_d1_top.csv", index=False, encoding="utf-8-sig")

    # same-word prob
    sw = sameword_prob(adj, inword, min_adj_freq=5)
    pd.DataFrame(sw[:500], columns=["x","y","f_adj","f_inword","p_sameword_given_xy"]).to_csv(
        "out/char_sameword_prob_top.csv", index=False, encoding="utf-8-sig"
    )

    # 也可以把 d=2..5 的共现表各导一个文件（报告里放1-2张就够）