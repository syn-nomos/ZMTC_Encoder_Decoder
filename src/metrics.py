from typing import List, Dict, Any, Set
from collections import defaultdict
import torch
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

@torch.no_grad()
def micro_f1(pred: List[List[str]], gold: List[List[str]]) -> Dict[str, float]:
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return {"micro_p": prec, "micro_r": rec, "micro_f1": f1}

def avg_predicted_labels(pred_label_indices: List[List[int]]) -> float:
    """Average number of predicted labels per sample in a batch.
    Assumes each inner list contains ONLY label indices (EOS excluded).
    """
    if not pred_label_indices:
        return 0.0
    return sum(len(seq) for seq in pred_label_indices) / float(len(pred_label_indices))

def format_avg_predicted_labels(pred_label_indices: List[List[int]]) -> str:
    avg_k = avg_predicted_labels(pred_label_indices)
    if not pred_label_indices:
        return "avgK=0.00 (min=0 max=0)"
    lens = [len(s) for s in pred_label_indices]
    return f"avgK={avg_k:.2f} (min={min(lens)} max={max(lens)})"



def make_mlb_from_idx2concept(idx2concept):
    # idx2concept: Dict[int, str]
    classes = [str(idx2concept[i]) for i in range(len(idx2concept))]
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit(classes)
    return mlb

@torch.no_grad()
def jaccard_score_multilabel(
    pred: List[List[str]],
    gold: List[List[str]],
    mlb: MultiLabelBinarizer,
) -> Dict[str, float]:
    """
    Sample-averaged Jaccard index for multi-label classification.
    """
    assert len(pred) == len(gold)


    Y_true = mlb.transform(gold)
    Y_pred = mlb.transform(pred)

    # average="samples" → per-instance Jaccard, then mean
    jacc = jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)

    return {"jaccard": float(jacc)}


@torch.no_grad()
def hamming_loss_multilabel(
    pred: List[List[str]],
    gold: List[List[str]],
    mlb: MultiLabelBinarizer,
) -> Dict[str, float]:
    """
    Hamming loss for multi-label classification.
    """
    assert len(pred) == len(gold)


    Y_true = mlb.transform(gold)
    Y_pred = mlb.transform(pred)

    hl = hamming_loss(Y_true, Y_pred)

    return {"hamming_loss": float(hl)}

@torch.no_grad()
def subset_accuracy(
    pred: List[List[str]],
    gold: List[List[str]],
) -> Dict[str, float]:
    """
    Subset Accuracy (Exact Match) for multi-label classification.

    pred, gold: lists of label lists (strings), one list per sample
    """
    assert len(pred) == len(gold)

    correct = 0
    for p, g in zip(pred, gold):
        if set(p) == set(g):
            correct += 1

    acc = correct / max(len(pred), 1)

    return {"subset_accuracy": acc}

@torch.no_grad()
def label_cardinality_error(
    pred: List[List[str]],
    gold: List[List[str]],
) -> Dict[str, float]:
    """
    Label Cardinality Error (absolute).

    Measures how far the number of predicted labels is
    from the number of gold labels.
    """
    assert len(pred) == len(gold)

    errors = []
    for p, g in zip(pred, gold):
        errors.append(abs(len(p) - len(g)))

    return {
        "cardinality_error": sum(errors) / max(len(errors), 1)
    }

def macro_f1_labelwise(
    pred: List[List[str]],
    gold: List[List[str]],
    label_space: Set[str] | None = None,
    ignore_labels_missing_in_gold: bool = True,
) -> Dict[str, Any]:
    eps = 1e-12

    labels_pred = set()
    labels_gold = set()
    for p in pred:
        labels_pred.update(p)
    for g in gold:
        labels_gold.update(g)

    all_labels = label_space if label_space is not None else (labels_pred | labels_gold)

    eval_labels = set(labels_gold) if ignore_labels_missing_in_gold else set(all_labels)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)

    for p_list, g_list in zip(pred, gold):
        p_set = set(p_list) & eval_labels
        g_set = set(g_list) & eval_labels

        for l in g_set:
            support[l] += 1

        for l in (p_set & g_set):
            tp[l] += 1
        for l in (p_set - g_set):
            fp[l] += 1
        for l in (g_set - p_set):
            fn[l] += 1

    ps, rs, f1s = [], [], []
    for l in eval_labels:
        p_l = tp[l] / (tp[l] + fp[l] + eps)
        r_l = tp[l] / (tp[l] + fn[l] + eps)
        f1_l = 0.0 if (p_l + r_l) <= 0 else (2 * p_l * r_l) / (p_l + r_l + eps)
        ps.append(p_l); rs.append(r_l); f1s.append(f1_l)

    denom = max(len(eval_labels), 1)
    return {
        "macro_p": sum(ps) / denom,
        "macro_r": sum(rs) / denom,
        "macro_f1": sum(f1s) / denom,
        "num_labels_averaged": len(eval_labels),
    }
