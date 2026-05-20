import re
from typing import List, Tuple, Iterable

SENT_END = {"。", "！", "？"}  # 只用这三个做句子边界

def _parse_bracket_token(chunks: List[str]) -> Tuple[str, str]:
    """
    chunks: 类似 ["[中央/n", "人民/n", "广播/vn", "电台/n]nt"]
    返回: ("中央人民广播电台", "nt")
    """
    last = chunks[-1]
    # last 形如 "电台/n]nt"
    m = re.search(r"\](\w+)$", last)
    if not m:
        raise ValueError(f"bad bracket token ending: {last}")
    outer_pos = m.group(1)

    words = []
    for i, ck in enumerate(chunks):
        # 去掉开头的 '['
        ck2 = ck[1:] if ck.startswith("[") else ck
        # 去掉结尾的 ']pos'
        if i == len(chunks) - 1:
            ck2 = re.sub(r"\]\w+$", "", ck2)
        # 现在 ck2 形如 "中央/n" 或 "电台/n"
        if "/" not in ck2:
            continue
        w, _inner_pos = ck2.rsplit("/", 1)
        words.append(w)

    return ("".join(words), outer_pos)

def parse_1998_line(line: str) -> List[Tuple[str, str]]:
    """
    输入一行：ID  token token token ...
    输出：[(word,pos), ...]
    """
    line = line.strip()
    if not line:
        return []

    parts = line.split()
    if len(parts) <= 1:
        return []

    tokens = []
    i = 1  # 跳过第 0 列 ID
    while i < len(parts):
        p = parts[i]

        # bracket merged token
        if p.startswith("["):
            j = i
            while j < len(parts) and ("]" not in parts[j]):
                j += 1
            if j >= len(parts):
                # 没闭合，兜底当普通 token 处理
                pass
            else:
                merged_word, merged_pos = _parse_bracket_token(parts[i:j+1])
                tokens.append((merged_word, merged_pos))
                i = j + 1
                continue

        # normal token: word/pos
        if "/" in p:
            w, pos = p.rsplit("/", 1)
            tokens.append((w, pos))
        i += 1

    return tokens

def iter_1998_sentences(path: str) -> Iterable[List[Tuple[str, str]]]:
    """
    读取大文件，按 。！？ 切分成句子（token 列表）
    """
    sent: List[Tuple[str, str]] = []
    with open(path, "r", encoding="gb18030", errors="strict") as f:
        for line in f:
            toks = parse_1998_line(line)
            for w, pos in toks:
                sent.append((w, pos))
                if pos == "w" and w in SENT_END:
                    yield sent
                    sent = []
    if sent:
        yield sent

if __name__ == "__main__":
    # 自测：打印前3句
    import itertools
    for s in itertools.islice(iter_1998_sentences("data/ChineseCorpus199801.txt"), 3):
        print(s)
        print("----")