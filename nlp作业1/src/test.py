import json
from itertools import islice

path = "out/rmrb_2026_01_clean.jsonl"  # 或未clean版也行
with open(path, "r", encoding="utf-8") as f:
    obj = json.loads(next(f))
    print("title:", obj.get("title","")[:80])
    t = obj.get("text","")
    print("text head:", t[:500])