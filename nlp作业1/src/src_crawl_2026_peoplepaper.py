import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time as _time

BASE = "https://paper.people.com.cn"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36")

# 目录页（版面页）规律：/rmrb/pc/layout/YYYYMM/DD/node_XX.html
LAYOUT_TMPL = BASE + "/rmrb/pc/layout/{yyyymm}/{dd}/node_{node:02d}.html"

# 文章页规律：/rmrb/pc/content/YYYYMM/DD/content_XXXXXXXX.html
# CONTENT_RE = re.compile(r"^/rmrb/pc/content/\d{6}/\d{2}/content_\d+\.html$")
CONTENT_RE = re.compile(r"/rmrb/pc/content/\d{6}/\d{2}/content_\d+\.html$") #改正则 使用相对路径


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

@dataclass
class Article:
    dt: str
    url: str
    title: str
    text: str

def daterange(d1: date, d2: date) -> Iterable[date]:
    d = d1
    while d <= d2:
        yield d
        d += timedelta(days=1)

def get_html(url: str, timeout=10, retries=2, sleep=0.3):
    for i in range(retries):
        t0 = _time.time()
        try:
            r = SESSION.get(url, timeout=timeout)
            dt = _time.time() - t0
            print(f"[{r.status_code}] {dt:.2f}s {url}")

            if r.status_code != 200:
                _time.sleep(sleep * (i + 1))
                continue

            # 编码修复（保留你之前那段）
            apparent = (r.apparent_encoding or "").lower()
            enc = (r.encoding or "").lower()
            if enc in ("iso-8859-1", ""):
                if "utf" in apparent:
                    r.encoding = "utf-8"
                elif "gb" in apparent or "18030" in apparent:
                    r.encoding = "gb18030"
                else:
                    r.encoding = "gb18030"

            html = r.text
            if html:
                _time.sleep(sleep)
                return html
        except requests.RequestException as e:
            dt = _time.time() - t0
            print(f"[ERR] {dt:.2f}s {url} -> {e}")
            _time.sleep(sleep * (i + 1))
    return None
# def extract_content_links_from_layout(html: str) -> List[str]:
#     soup = BeautifulSoup(html, "lxml")
#     links = []
#     for a in soup.select("a[href]"):
#         href = a.get("href", "").strip()
#         if CONTENT_RE.match(href):
#             links.append(BASE + href)
#     return links

def extract_content_links_from_layout(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href or href == "#":
            continue

        abs_url = urljoin(base_url, href)
        abs_url = abs_url.split("#", 1)[0].split("?", 1)[0]  # 去掉锚点/参数便于匹配与去重

        if CONTENT_RE.search(abs_url):
            links.append(abs_url)

    return links

def clean_text(s: str) -> str:
    # 简单清洗：去多余空白行
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def parse_article_page(html: str, url: str, dt: str) -> Optional[Article]:
    soup = BeautifulSoup(html, "lxml")

    # 标题：页面通常在 <title> 或正文附近有明显标题
    title = soup.title.get_text(strip=True) if soup.title else ""
    # title 往往带网站后缀，这里做个弱清洗
    title = title.replace("_人民日报数字报", "").strip()

    # 正文提取策略（通用版）：
    # 1) 优先找文章主体区域：从页面文本中抽取“目录/上一篇/下一篇”之后的大段正文
    # 2) 如果失败，退化为 body.get_text()
    body_text = soup.get_text("\n", strip=True)
    body_text = clean_text(body_text)

    # 经验切割：文章页会出现“返回目录”“上一篇”“下一篇”等导航词，
    # 正文通常在这些之后开始。我们用这些词做一个尽量稳的切分。
    nav_markers = ["返回目录", "上一篇", "下一篇", "全文复制"]
    cut_pos = -1
    for mk in nav_markers:
        p = body_text.find(mk)
        if p != -1:
            cut_pos = max(cut_pos, p)
    if cut_pos != -1:
        text = body_text[cut_pos:]
        # 再把导航词本身去掉（保守处理）
        for mk in nav_markers:
            text = text.replace(mk, "")
        text = clean_text(text)
    else:
        text = body_text

    # 过滤掉明显抓错的页面
    if len(text) < 200:
        return None

    return Article(dt=dt, url=url, title=title, text=text)

def crawl_month_2026_01(out_path="out/rmrb_2026_01.jsonl",
                        start=date(2026,1,1), end=date(2026,1,31),
                        max_node=10):
    seen_article_urls: Set[str] = set()
    all_articles: List[Article] = []

    for d in tqdm(list(daterange(start, end)), desc="days"):
        yyyymm = f"{d.year}{d.month:02d}"
        dd = f"{d.day:02d}"
        dt_str = f"{d.year}-{d.month:02d}-{d.day:02d}"

        # 逐个 node 抓（不同日期版面数不同；抓到 404/无内容就跳过）
        content_urls_today: Set[str] = set()
        for node in range(1, max_node+1):
            layout_url = LAYOUT_TMPL.format(yyyymm=yyyymm, dd=dd, node=node)
            html = get_html(layout_url)
            if not html:
                continue
            links = extract_content_links_from_layout(html, layout_url)
            #print(layout_url, "links=", len(links))
            for u in links:
                content_urls_today.add(u)

        # 抓文章页
        for url in sorted(content_urls_today):
            if url in seen_article_urls:
                continue
            html = get_html(url)
            if not html:
                continue
            art = parse_article_page(html, url=url, dt=dt_str)
            if not art:
                continue
            seen_article_urls.add(url)
            all_articles.append(art)

    # 写 jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for a in all_articles:
            f.write(json.dumps(a.__dict__, ensure_ascii=False) + "\n")

    print(f"saved: {out_path}, articles={len(all_articles)}")

if __name__ == "__main__":

    test_url = "https://paper.people.com.cn/rmrb/pc/layout/202601/01/node_01.html"
    html = get_html(test_url, retries=1, sleep=0)
    print("layout html len:", 0 if not html else len(html))
    crawl_month_2026_01()