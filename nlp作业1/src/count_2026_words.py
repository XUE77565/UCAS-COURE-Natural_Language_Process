import json
import re
from collections import Counter
import jieba
import pandas as pd

PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

def is_good_token(w: str) -> bool:
    w = w.strip()
    if not w:
        return False
    if PUNCT_RE.match(w):
        return False
    if w.isdigit():
        return False
    return True

def count_2026_words(
    jsonl_path: str = "out/rmrb_2026_01_clean.jsonl",
    out_csv: str = "out/2026_word_freq.csv",
    min_len: int = 1,
):
    c = Counter()
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "") or ""
            for w in jieba.lcut(text):
                if not is_good_token(w):
                    continue
                if len(w) < min_len:
                    continue
                c[w] += 1
                total += 1

    df = pd.DataFrame(c.most_common(), columns=["word", "freq"])
    df["p"] = df["freq"] / max(total, 1)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved:", out_csv, "unique_words=", len(df), "total_tokens=", total)

if __name__ == "__main__":
    count_2026_words()