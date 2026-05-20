import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import *
from src.data import load_and_clean_data, build_vocab, build_ngram_data
from src.models import FNNModel, RNNModel, LSTMModel

def train(model_name="FNN"):
    # ====== 数据处理 ======
    words = load_and_clean_data(CORPUS_PATH, ENCODING)
    vocab, word_to_idx, idx_to_word = build_vocab(words, MIN_FREQ)
    data = build_ngram_data(words, word_to_idx, CONTEXT_SIZE)

    print(f"词表大小: {len(vocab)}")
    print(f"训练样本数: {len(data)}")

    inputs = torch.tensor([x[0] for x in data], dtype=torch.long)
    targets = torch.tensor([x[1] for x in data], dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ====== 选择模型 ======
    if model_name == "FNN":
        model = FNNModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
    elif model_name == "RNN":
        model = RNNModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)
    else:
        model = LSTMModel(len(vocab), EMBEDDING_DIM, HIDDEN_DIM)

    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ====== 训练 ======
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_inputs, batch_targets in loader:
            batch_inputs, batch_targets = batch_inputs.to(DEVICE), batch_targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{model_name} Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(loader):.4f}")

    # ====== 保存词向量 ======
    embeddings = model.embeddings.weight.data.cpu()
    torch.save({
        "embeddings": embeddings,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word
    }, f"{model_name}_embeddings.pt")
    print(f"✅ 已保存 {model_name} 词向量")

if __name__ == "__main__":
    train("FNN")
    train("RNN")
    train("LSTM")