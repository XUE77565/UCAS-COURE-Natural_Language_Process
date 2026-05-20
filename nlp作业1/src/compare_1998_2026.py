import math
import pandas as pd

def compare(
    freq_1998_csv: str = "out/1998_word_freq.csv",
    freq_2026_csv: str = "out/2026_word_freq.csv",
    out_top_2026: str = "out/top_2026_over_1998.csv",
    out_top_1998: str = "out/top_1998_over_2026.csv",
    out_new: str = "out/new_words_2026.csv",
    eps: float = 1e-9,
    topn: int = 300,
    new_min_freq_2026: int = 5,
    new_min_len: int = 2,
):
    d98 = pd.read_csv(freq_1998_csv)
    d26 = pd.read_csv(freq_2026_csv)

    d98 = d98[["word", "freq", "p"]].rename(columns={"freq": "freq_1998", "p": "p_1998"})
    d26 = d26[["word", "freq", "p"]].rename(columns={"freq": "freq_2026", "p": "p_2026"})

    merged = pd.merge(d26, d98, on="word", how="outer").fillna(0)

    # log ratio：越大越偏向2026，越小越偏向1998
    merged["logratio_2026_vs_1998"] = merged.apply(
        lambda r: math.log((r["p_2026"] + eps) / (r["p_1998"] + eps)), axis=1
    )
    merged["delta_p"] = merged["p_2026"] - merged["p_1998"]

    # Top: 2026 over 1998
    top_2026 = merged.sort_values("logratio_2026_vs_1998", ascending=False).head(topn)
    top_2026.to_csv(out_top_2026, index=False, encoding="utf-8-sig")

    # Top: 1998 over 2026
    top_1998 = merged.sort_values("logratio_2026_vs_1998", ascending=True).head(topn)
    top_1998.to_csv(out_top_1998, index=False, encoding="utf-8-sig")

    # New words: freq_2026 >= m and freq_1998 == 0
    new_words = merged[
        (merged["freq_2026"] >= new_min_freq_2026) &
        (merged["freq_1998"] == 0) &
        (merged["word"].astype(str).str.len() >= new_min_len)
    ].sort_values(["freq_2026", "p_2026"], ascending=False)

    new_words.to_csv(out_new, index=False, encoding="utf-8-sig")

    print("saved:", out_top_2026)
    print("saved:", out_top_1998)
    print("saved:", out_new)
    print("new_words_count:", len(new_words))

if __name__ == "__main__":
    compare()