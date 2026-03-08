import torch
from torch.utils.data import DataLoader
from typing import Dict
from typing import List, Dict
from utils import build_idx2concept, compute_k_statistics
from metrics import macro_f1_labelwise,micro_f1,label_cardinality_error,subset_accuracy,jaccard_score_multilabel,hamming_loss_multilabel

@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    label_mem: torch.Tensor,
    concept2idx: Dict[str, int],
    device: torch.device,
    max_steps: int = 32,
    runBeamSearch: bool = False,
    mlb = None
) -> Dict[str, float]:
    model.eval()
    idx2concept = build_idx2concept(concept2idx)

    total_loss = 0.0
    n_batches = 0
    total_pred_k = 0.0
    total_samples = 0

    all_pred_concepts: List[List[str]] = []
    all_gold_concepts: List[List[str]] = []
    # ---- K statistics accumulators (epoch-level) ----
    sum_pred_k = 0.0
    sum_gold_k = 0.0
    sum_gold_k_hat = 0.0
    sum_k_hat = 0.0

    for batch in dataloader:
        batch = batch.to(device)

        # 1) LOSS: teacher forcing με gold indices
        out = model(
            doc_input_ids=batch.doc_input_ids,
            doc_attention_mask=batch.doc_attention_mask,
            dec_prev_indices=batch.dec_prev_indices,
            dec_attention_mask=batch.dec_attention_mask,
            label_mem=label_mem,
            tgt_label_indices=batch.tgt_indices,
            tgt_pad_mask=batch.tgt_pad_mask,
        )
        loss = out["loss"]
        total_loss += float(loss.item())
        n_batches += 1

        # 2) METRICS: greedy generation -> indices -> concept ids
        if not runBeamSearch:
          pred_idxs = model.generate_greedy(
              doc_input_ids=batch.doc_input_ids,
              doc_attention_mask=batch.doc_attention_mask,
              label_mem=label_mem,
              max_new_tokens=max_steps,
              no_repeat_labels=True,
              min_labels=None,
          )
        else:
          pred_idxs = model.generate_beam(
              doc_input_ids=batch.doc_input_ids,
              doc_attention_mask=batch.doc_attention_mask,
              label_mem=label_mem,
              beam_size=5,
              max_new_tokens=max_steps,
              no_repeat_labels=True,
              min_labels=None,
              length_penalty=0.2,
          )


        total_pred_k += sum(len(s) for s in pred_idxs)
        total_samples += len(pred_idxs)
        pred_concepts = [
            {idx2concept[i] for i in preds}
            for preds in pred_idxs
        ]

        gold_concepts = [
            set(item["concepts"] if "concepts" in item.keys() else item["labels"])
            for item in batch.raw_items
        ]
        stats = compute_k_statistics(
                pred_indices=pred_idxs,
                gold_concepts=gold_concepts,
                count_logits=out.get("count_logits"),
                max_pred_labels=model.max_pred_labels,
        )
        B = len(pred_idxs)
        sum_pred_k += stats["avg_pred_k"] * B
        sum_gold_k += stats["avg_gold_k"] * B

        if stats["avg_k_hat"] is not None:
            sum_k_hat += stats["avg_k_hat"] * B

        if stats["avg_gold_k_hat"] is not None:
            sum_gold_k_hat += stats["avg_gold_k_hat"] * B

        # pred_concepts = preds_indices_to_concepts(pred_idxs, idx2concept)
        # gold_concepts = gold_items_to_concepts(batch.raw_items)


        all_pred_concepts.extend(pred_concepts)
        all_gold_concepts.extend(gold_concepts)
        metrics_sub_acc = subset_accuracy(pred_concepts, gold_concepts)
        cardinality_error = label_cardinality_error(pred_concepts, gold_concepts)
        card_err += cardinality_error.get("cardinality_error")
        sub_acc += metrics_sub_acc.get("subset_accuracy")
    cardinality_error["cardinality_error"] = card_err/total_samples
    metrics_sub_acc["subset_accuracy"] = sub_acc/total_samples
    metrics_micro = micro_f1(all_pred_concepts, all_gold_concepts)
    metrics_macro = macro_f1_labelwise(
        all_pred_concepts,
        all_gold_concepts,
        label_space=set(concept2idx.keys()),
        ignore_labels_missing_in_gold=True,
    )
    #metrics_macro = macro_f1_labelwise(all_pred_concepts, all_gold_concepts)
    #metrics_sub_acc = subset_accuracy(pred_concepts, gold_concepts)
    #cardinality_error = label_cardinality_error(pred_concepts, gold_concepts)
    if mlb is not None:
    # Calculate Jaccard score
      metrics_jaccard = jaccard_score_multilabel(all_pred_concepts, all_gold_concepts,mlb)
      metrics_hamming = hamming_loss_multilabel(all_pred_concepts, all_gold_concepts,mlb)
    if mlb is not None:
      metrics = {**metrics_micro, **metrics_macro, **metrics_jaccard,**metrics_hamming, **metrics_sub_acc, **cardinality_error}
    else:
      metrics = {**metrics_micro, **metrics_macro, **metrics_sub_acc, **cardinality_error}
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    metrics["avg_pred_k"] = sum_pred_k / total_samples
    metrics["avg_gold_k"] = sum_gold_k / total_samples

    metrics["avg_k_hat"] = (sum_k_hat / total_samples) if sum_k_hat > 0 else None
    metrics["avg_gold_k_hat"] = (sum_gold_k_hat / total_samples) if sum_gold_k_hat > 0 else None
    # print("logit_scale(exp) =", float(model.logit_scale.exp().item()))
    return metrics