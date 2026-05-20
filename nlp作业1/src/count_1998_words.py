from collections import Counter
import pandas as pd
from parse_1998 import iter_1998_sentences

def count_1998_words(
    path_1998: str = "data/ChineseCorpus199801.txt",
    out_csv: str = "out/1998_word_freq.csv",
    drop_pos_w: bool = True,
    drop_pos_t: bool = False,   # 可选：是否过滤时间词 t
    min_len: int = 1,
):
    c = Counter()
    total = 0

    for sent in iter_1998_sentences(path_1998):
        for w, pos in sent:
            if drop_pos_w and pos == "w":
                continue
            if drop_pos_t and pos == "t":
                continue
            w = w.strip()
            if len(w) < min_len:
                continue
            c[w] += 1
            total += 1

    df = pd.DataFrame(c.most_common(), columns=["word", "freq"])
    df["p"] = df["freq"] / max(total, 1)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv, "unique_words=", len(df), "total_tokens=", total)

if __name__ == "__main__":
    count_1998_words()