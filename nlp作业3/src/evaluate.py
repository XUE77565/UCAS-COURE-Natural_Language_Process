def get_word_spans(words):
    """
    将分词列表转换为字符区间集合，用于计算 P, R, F1
    例如：['中国', '人民'] -> {(0, 2), (2, 4)}
    """
    spans = set()
    start = 0
    for word in words:
        end = start + len(word)
        spans.add((start, end))
        start = end
    return spans

def calculate_prf(pred_words, gt_words):
    """
    计算精准率(Precision)、召回率(Recall)和F1值
    """
    pred_spans = get_word_spans(pred_words)
    gt_spans = get_word_spans(gt_words)
    
    correct_spans = pred_spans.intersection(gt_spans)
    correct = len(correct_spans)
    
    precision = correct / len(pred_spans) if len(pred_spans) > 0 else 0.0
    recall = correct / len(gt_spans) if len(gt_spans) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_dataset(predictions, ground_truths):
    """
    计算整个数据集的平均 P, R, F1
    predictions: list of list of strings (模型的预测分词结果列表)
    ground_truths: list of list of strings (标准答案分词结果列表)
    """
    total_p, total_r, total_f1 = 0, 0, 0
    valid_count = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if not gt:
            continue
        p, r, f1 = calculate_prf(pred, gt)
        total_p += p
        total_r += r
        total_f1 += f1
        valid_count += 1
        
    if valid_count == 0:
        return 0, 0, 0
        
    avg_p = total_p / valid_count
    avg_r = total_r / valid_count
    avg_f1 = total_f1 / valid_count
    
    return avg_p, avg_r, avg_f1
