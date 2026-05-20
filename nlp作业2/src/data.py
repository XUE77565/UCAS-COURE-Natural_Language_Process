import re
from collections import Counter

def load_and_clean_data(file_path, encoding="utf-8"):
    words = []
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                word = token.split("/")[0]
                if re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', word):
                    words.append(word)
    return words

def build_vocab(words, min_freq=3):
    counts = Counter(words)
    vocab = [w for w, c in counts.items() if c >= min_freq]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

def build_ngram_data(words, word_to_idx, context_size=2):
    data = []
    for i in range(len(words) - context_size):
        context = words[i:i+context_size]
        target = words[i+context_size]
        if all(w in word_to_idx for w in context) and target in word_to_idx:
            context_idx = [word_to_idx[w] for w in context]
            target_idx = word_to_idx[target]
            data.append((context_idx, target_idx))
    return data