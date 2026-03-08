from typing import Any
from typing import Optional, List, Dict, Any, Tuple,Set, Iterable,Union
import json
import re
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

def build_document_text(ex: Dict[str, Any]) -> str:
    """
    dev.jsonl sample has:
      title (str), header (str), recitals (str), main_body (list[str]), attachments (str)

    Join them into one text.
    """
    parts = []
    for k in ["title", "header", "recitals","text"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.lower().strip())

    mb = ex.get("main_body")
    if isinstance(mb, list):
        mb_text = " ".join([s.strip() for s in mb if isinstance(s, str) and s.strip()])
        if mb_text:
            parts.append(mb_text.lower())

    # att = ex.get("attachments")
    # if isinstance(att, str) and att.strip():
    #     parts.append(att.strip())

    return " ".join(parts)


def compute_label_stats_from_splits(
    dataset_jsonl_paths: List[str],
    concepts_field: str = "labels",
) -> Tuple[int, int, float]:
    """
    Computes label-per-document statistics across all splits.

    Returns:
      min_labels_per_doc,
      max_labels_per_doc,
      avg_labels_per_doc
    """
    counts: List[int] = []

    for p in dataset_jsonl_paths:
        for row in iter_jsonl(p):
            labels = row.get(concepts_field, []) or []
            counts.append(len(labels))

    if not counts:
        return 0, 0, 0.0

    mn = min(counts)
    mx = max(counts)
    avg = sum(counts) / len(counts)

    return mn, mx, avg

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def concept_sort_key(concept_id: str):
    """
    'Ascending by id' for mixed ids:
    - numeric strings first (sorted by int)
    - then non-numeric (sorted lexicographically)
    """
    s = str(concept_id)
    if re.fullmatch(r"\d+", s):
        return (0, int(s))
    return (1, s)
def collect_used_concepts_from_splits(
    dataset_jsonl_paths: List[str],
    concepts_field: str = "concepts",
) -> Set[str]:
    """
    Collect union of concept IDs used across splits.
    Assumes each row has a list field like row["concepts"].
    """
    used: Set[str] = set()
    for p in dataset_jsonl_paths:
        for row in iter_jsonl(p):
            for c in row.get(concepts_field, []) or []:
                used.add(str(c))
    return used
def load_label_descriptions(
    labels_jsonl_path: str,
    *,
    only_used_labels: bool = False,
    used_concepts: Optional[Set[str]] = None,
    dataset_jsonl_paths: Optional[List[str]] = None,
    concepts_field: str = "concepts",#"concepts", "labels"
    keep_missing_desc: bool = True,
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Reads gpt_labels_descriptions_eurlex.jsonl.

    If only_used_labels=True, it will restrict the label space to concepts
    that appear in the dataset (train/dev/test union).

    Provide either:
      - used_concepts (preferred), OR
      - dataset_jsonl_paths (it will collect used concepts from those splits)

    Returns:
      sorted_concepts: list of concept_id sorted ascending by concept_sort_key
      concept2idx: concept_id -> idx
      idx2concept: idx -> concept_id
      concept2desc: concept_id -> description_text (LLM_Response_text)
    """
    concept2desc: Dict[str, str] = {}
    for row in iter_jsonl(labels_jsonl_path):
        cid = str(row["concept_id"])
        concept2desc[cid] = row.get("LLM_Response_text", "").lower() or "" #LLM_Response_text

    # --- decide used concepts ---
    if only_used_labels:
        if used_concepts is None:
            if not dataset_jsonl_paths:
                raise ValueError(
                    "only_used_labels=True requires either used_concepts or dataset_jsonl_paths."
                )
            used_concepts = collect_used_concepts_from_splits(
                dataset_jsonl_paths=dataset_jsonl_paths,
                concepts_field=concepts_field,
            )

        # filter label space
        filtered = {}
        for cid in used_concepts:
            if cid in concept2desc:
                filtered[cid] = concept2desc[cid]
            else:
                # concept appears in dataset but no description entry exists
                if keep_missing_desc:
                    filtered[cid] = ""  # keep it with empty desc
        concept2desc = filtered

    sorted_concepts = sorted(concept2desc.keys(), key=concept_sort_key)
    concept2idx = {cid: i for i, cid in enumerate(sorted_concepts)}
    idx2concept = {i: cid for cid, i in concept2idx.items()}

    return sorted_concepts, concept2idx, idx2concept, concept2desc

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
@torch.no_grad()
def precompute_label_memory(
    labels_text: list[str],
    tokenizer,
    encoder_model,
    device,
    batch_size: int = 64,
    max_length: int = 32,
    pooling: str = "cls",   # "cls" ή "mean"
    normalize: bool = True,
):
    encoder_model.eval()
    all_vecs = []

    for i in range(0, len(labels_text), batch_size):
        batch = labels_text[i:i+batch_size]
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        out = encoder_model(**tok, return_dict=True)
        h = out.last_hidden_state  # [B,T,H]

        if pooling == "cls":
            vec = h[:, 0, :]  # [B,H]
        elif pooling == "mean":
            vec = mean_pooling(out,tok["attention_mask"])
        else:
            vec = last_token_pool(h,tok["attention_mask"])

        if normalize:
            vec = F.normalize(vec, p=2, dim=-1)

        all_vecs.append(vec.detach().cpu())

    label_mem = torch.cat(all_vecs, dim=0).to(device)  # [L,d_model]
    return label_mem

def build_idx2concept(concept2idx: Dict[str, int]) -> List[str]:
    idx2concept = [None] * len(concept2idx)
    for c, i in concept2idx.items():
        idx2concept[i] = c
    return idx2concept

def preds_indices_to_concepts(pred_idxs: List[List[int]], idx2concept: List[str]) -> List[List[str]]:
    out = []
    n = len(idx2concept)
    for seq in pred_idxs:
        seen = set()
        cs = []
        for i in seq:
            if 0 <= i < n:
                cid = idx2concept[i]
                if cid not in seen:
                    seen.add(cid)
                    cs.append(cid)
        out.append(cs)
    return out

def gold_items_to_concepts(raw_items: List[Dict[str, Any]], concept2idx: Dict[str, int]) -> List[List[str]]:
    out = []
    for it in raw_items:
        cs = [str(c) for c in it.get("concepts", [])]
        cs = [c for c in cs if c in concept2idx]
        out.append(list(set(cs)))
    return out

def compute_k_statistics(
    pred_indices: List[List[int]],
    gold_concepts: List[Union[List[Any], set]],
    count_logits: Optional[torch.Tensor] = None,
    max_pred_labels: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes:
        - avg_pred_k
        - avg_gold_k
        - avg_gold_k_hat (clamped gold K)
        - avg_k_hat (from count_logits argmax, if provided)

    Args:
        pred_indices: list of predicted label index lists (len B)
        gold_concepts: list of gold label sets/lists (len B)
        count_logits: optional tensor [B, maxK+1] from count_head
        max_pred_labels: required if you want gold_k_hat

    Returns:
        dict with statistics
    """

    B = len(pred_indices)
    assert B == len(gold_concepts), "predictions and gold must have same batch size"

    # ---- pred_k ----
    pred_k = torch.tensor([len(p) for p in pred_indices], dtype=torch.float32)
    avg_pred_k = float(pred_k.mean().item())

    # ---- gold_k ----
    gold_k = torch.tensor([len(g) for g in gold_concepts], dtype=torch.float32)
    avg_gold_k = float(gold_k.mean().item())

    # ---- gold_k_hat (clamped) ----
    if max_pred_labels is not None:
        gold_k_hat = gold_k.clamp(min=0, max=float(max_pred_labels))
        avg_gold_k_hat = float(gold_k_hat.mean().item())
    else:
        avg_gold_k_hat = None

    # ---- k_hat from count_head ----
    if count_logits is not None:
        with torch.no_grad():
            k_hat = torch.argmax(count_logits, dim=-1).float()  # [B]
            avg_k_hat = float(k_hat.mean().item())
    else:
        avg_k_hat = None

    return {
        "avg_pred_k": avg_pred_k,
        "avg_gold_k": avg_gold_k,
        "avg_gold_k_hat": avg_gold_k_hat,
        "avg_k_hat": avg_k_hat,
    }

def get_transformer_layers(hf_encoder) -> nn.ModuleList:
    """
    Works for common HF encoders:
      - DistilBERT: transformer.layer
      - MPNet/BERT/RoBERTa/DeBERTa: encoder.layer
      - Some: encoder.layers or layers
    """
    candidates = [
        ("transformer", "layer"),         # DistilBERT
        ("encoder", "layer"),             # MPNet/BERT/RoBERTa
        ("encoder", "layers"),            # some variants
        ("layer",),
        ("layers",),                      # fallback
    ]

    for path in candidates:
        obj = hf_encoder
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj

    raise AttributeError(
        f"Cannot locate transformer layers for encoder: {hf_encoder.__class__.__name__}. "
        "Tried transformer.layer, encoder.layer, encoder.layers, layers."
    )

def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

def unfreeze_encoder(model, unfreeze: str = "last", unfreeze_layer_norms: bool = True):
    """
    unfreeze:
      - "last": unfreeze only the last transformer block (+ optional LayerNorms)
      - "all":  unfreeze all encoder params
    """
    # freeze all first
    for p in model.encoder.parameters():
        p.requires_grad = False

    if unfreeze == "all":
        for p in model.encoder.parameters():
            p.requires_grad = True
    else:
        layers = get_transformer_layers(model.encoder)
        for layer in layers[-4:]:
            for p in layer.parameters():
                p.requires_grad = True

        if unfreeze_layer_norms:
            for m in model.encoder.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad = True

    model.encoder.train()
def unfreeze_encoder_last4_only(model):
    enc = model.encoder

    # freeze everything
    for p in enc.parameters():
        p.requires_grad = False

    layers = get_transformer_layers(enc)   # should return enc.encoder.layer for XLM-R
    last4 = layers[-4:]

    # unfreeze last 4 blocks (including their internal LNs)
    for block in last4:
        for p in block.parameters():
            p.requires_grad = True

    # optional: unfreeze final encoder layer norm (if present)
    # XLM-R has enc.encoder.layer_norm in some variants; safe guard:
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer_norm"):
        for p in enc.encoder.layer_norm.parameters():
            p.requires_grad = True

    enc.train()

def scan_label_count_stats(path: str) -> Tuple[int, float]:
    """
    Returns (max_labels, avg_labels) in the split.
    Useful to pick max_new_tokens.
    """
    n = 0
    s = 0
    mx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            k = 0
            if "concepts" in ex.keys():
              k = len(ex.get("concepts", []))
            elif "labels" in ex.keys():
              k = len(ex.get("labels", []))
            n += 1
            s += k
            mx = max(mx, k)
    return mx, (s / max(n, 1))



def _count_params(params):
    return sum(p.numel() for p in params)

def _count_trainable(params):
    return sum(p.numel() for p in params if p.requires_grad)

def _fmt_int(n: int) -> str:
    return f"{n:,}"

def print_trainable_parameter_summary(model: nn.Module):
    """
    Prints a compact summary in the exact style you showed, tailored for MPNetT5PointerLabelDecoder.
    Buckets:
      - encoder(mpnet)
      - enc_proj
      - decoder
      - decoder_embeddings (dec_embed)
      - lm_head (not present in pointer model -> 0)
      - other (heads, norms, etc.)
    """
    total_params = _count_params(model.parameters())
    trainable_params = _count_trainable(model.parameters())

    # buckets by name-prefix (robust for your pointer model)
    encoder_params = []
    enc_proj_params = []
    decoder_params = []
    dec_embed_params = []
    lm_head_params = []  # pointer model: none, but keep for format
    other_params = []

    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            encoder_params.append(p)
        elif name.startswith("enc_proj.") or name.startswith("enc_ln."):
            enc_proj_params.append(p)
        elif name.startswith("decoder."):
            decoder_params.append(p)
        elif name.startswith("dec_embed."):
            dec_embed_params.append(p)
        elif name.startswith("lm_head."):
            lm_head_params.append(p)
        else:
            other_params.append(p)

    buckets = {
        "encoder": _count_trainable(encoder_params),
        "enc_proj": _count_trainable(enc_proj_params),
        "decoder": _count_trainable(decoder_params),
        "decoder_embeddings": _count_trainable(dec_embed_params),
        "lm_head": _count_trainable(lm_head_params),
        "other": _count_trainable(other_params),
    }

    # pretty print (match your format)
    print("========== Trainable parameter summary ==========")
    print(f"Total params     : {_fmt_int(total_params)}")
    pct = (100.0 * trainable_params / total_params) if total_params else 0.0
    print(f"Trainable params : {_fmt_int(trainable_params)} ({pct:.2f}%)")
    print("------------------------------------------------")
    for k in ["encoder", "enc_proj", "decoder", "decoder_embeddings", "lm_head", "other"]:
        print(f"{k:<24s}: {_fmt_int(buckets[k])}")
    print("================================================")


def print_decoder_blocks_trainability(model: nn.Module):
    """
    Prints per-block trainability for T5Stack decoder blocks:
      Block 00: TRAINABLE (x/y params)
    Works for your MPNetT5PointerLabelDecoder where decoder is T5Stack.
    """
    if not hasattr(model, "decoder") or not hasattr(model.decoder, "block"):
        print("Model has no decoder.block; cannot print block trainability.")
        return

    blocks = model.decoder.block
    print("====== Decoder blocks trainability ======")
    for i, block in enumerate(blocks):
        params = list(block.parameters())
        n_total = len(params)
        n_train = sum(1 for p in params if p.requires_grad)
        status = "TRAINABLE" if n_train > 0 else "FROZEN"
        print(f"Block {i:02d}: {status} ({n_train}/{n_total} params)")
    print("=========================================")


def print_trainable_parameters_detailed(model: nn.Module, topk: int = 200):
    """
    Optional: detailed list of trainable tensors (largest first), similar to your current function.
    """
    total = 0
    trainable = 0
    rows = []

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            rows.append((n, name, tuple(p.shape), str(p.dtype), str(p.device)))

    rows.sort(reverse=True, key=lambda x: x[0])

    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    print("-" * 110)
    print(f"{'#params':>12s}  {'dtype':>10s}  {'device':>8s}  {'shape':>20s}  name")
    print("-" * 110)

    for n, name, shape, dtype, device in rows[:topk]:
        print(f"{n:12,d}  {dtype:>10s}  {device:>8s}  {str(shape):>20s}  {name}")

    if len(rows) > topk:
        print(f"... ({len(rows) - topk} more trainable tensors)")

def collect_concepts_from_items(items: List[Dict[str, Any]], key: str = "concepts") -> Set[str]:
    s: Set[str] = set()
    for it in items:
        for c in it.get(key, []) or []:
            s.add(str(c))
    return s

def collect_concepts_from_splits(
    train_items: List[Dict[str, Any]],
    dev_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    key: str = "concepts",
) -> Set[str]:
    return (
        collect_concepts_from_items(train_items, key)
        | collect_concepts_from_items(dev_items, key)
        | collect_concepts_from_items(test_items, key)
    )

import torch
from collections import OrderedDict

def load_oflan_tf_into_encoder(encoder_hf_model, ckpt_path: str, strict: bool = True):
    """
    Loads ONLY the OF-LAN document transformer weights (tf.*) into a HuggingFace AutoModel.

    encoder_hf_model: your AutoModel instance (e.g., DistilBERT/MPNet/etc)
    ckpt_path: path to OF-LAN checkpoint saved by the repo
    strict: if True, enforce exact match for transformer weights
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # support common checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    # filter keys for OF-LAN doc encoder
    tf_sd = OrderedDict()
    for k, v in sd.items():
        # common cases: "tf.xxx" or "module.tf.xxx"
        if k.startswith("tf."):
            tf_sd[k[len("tf."):]] = v
        elif k.startswith("module.tf."):
            tf_sd[k[len("module.tf."):]] = v

    if len(tf_sd) == 0:
        # helpful debug: show a few keys
        sample_keys = list(sd.keys())[:50]
        raise ValueError(
            "No keys matched tf.* or module.tf.* in checkpoint. "
            f"Sample keys: {sample_keys}"
        )

    missing, unexpected = encoder_hf_model.load_state_dict(tf_sd, strict=strict)
    return missing, unexpected

