import json
import re
import unicodedata
from typing import Dict, Any

# 常见脏字符：BOM、零宽字符
ZERO_WIDTH_AND_BOM = {
    "\ufeff",  # BOM
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
}

# 常见“看起来像空格”的字符
SPACE_LIKE = {
    "\u00a0",  # NBSP
    "\u3000",  # 全角空格
}

# 控制符（保留 \n \t）
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # 1) Unicode 兼容规范化：把一些兼容字符（全角/半角等）统一
    text = unicodedata.normalize("NFKC", text)

    # 2) 去 BOM / 零宽字符
    for ch in ZERO_WIDTH_AND_BOM:
        text = text.replace(ch, "")

    # 3) 空白归一（NBSP、全角空格 -> 普通空格）
    for ch in SPACE_LIKE:
        text = text.replace(ch, " ")

    # 4) 去控制符（保留换行/制表符）
    text = CONTROL_CHAR_RE.sub("", text)

    # 5) 合并空格/空行
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def clean_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    # 尽量保留原字段，只清洗常用的几个
    if "title" in obj and isinstance(obj["title"], str):
        obj["title"] = normalize_text(obj["title"])
    if "text" in obj and isinstance(obj["text"], str):
        obj["text"] = normalize_text(obj["text"])
    if "url" in obj and isinstance(obj["url"], str):
        obj["url"] = obj["url"].strip()
    if "dt" in obj and isinstance(obj["dt"], str):
        obj["dt"] = obj["dt"].strip()
    return obj

def clean_jsonl(
    in_path: str = "out/rmrb_2026_01.jsonl",
    out_path: str = "out/rmrb_2026_01_clean.jsonl",
    min_text_len: int = 200,
):
    n_in = 0
    n_out = 0
    n_drop_short = 0
    n_bad_json = 0

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue

            obj = clean_record(obj)

            text = obj.get("text", "")
            if isinstance(text, str) and len(text) < min_text_len:
                n_drop_short += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"input lines: {n_in}")
    print(f"bad json: {n_bad_json}")
    print(f"dropped(short text<{min_text_len}): {n_drop_short}")
    print(f"output lines: {n_out}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    clean_jsonl()