import os, time, json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from eval import evaluate
import torch
from torch.optim import AdamW
from transformers import Adafactor, get_linear_schedule_with_warmup

# -----------------------------
# CONFIG / FLAGS
# -----------------------------
DEFAULT_FREEZE_EPOCH = 2  # freeze starting at the BEGINNING of this epoch (e.g., epoch==10 -> freeze before epoch 10 loop)

# -----------------------------
# STAGE HELPERS
# -----------------------------

# def count_trainable(prefix):
#     tot = tr = 0
#     for n,p in model.named_parameters():
#         if n.startswith(prefix):
#             tot += p.numel()
#             tr += p.numel() if p.requires_grad else 0
#     return tr, tot



def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def get_stage(model) -> str:
    # If ANY encoder param is trainable -> "unfrozen"
    enc_params = list(model.encoder.parameters())
    if len(enc_params) == 0:
        return "unfrozen"
    return "unfrozen" if any(p.requires_grad for p in enc_params) else "frozen"

def should_be_frozen(epoch: int, freeze_enabled: bool, freeze_epoch: int) -> bool:
    # "freeze from epoch freeze_epoch and after"
    return bool(freeze_enabled and epoch >= int(freeze_epoch))

# -----------------------------
# OPTIM / SCHED BUILDERS
# -----------------------------
def ensure_initial_lr(optimizer):
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]
def build_optimizer(model, optimizer_name: str, lr: float, weight_decay: float):
    params = get_trainable_params(model)
    if optimizer_name.lower() == "adafactor":
        return Adafactor(
            params,
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
    return AdamW(params, lr=lr, weight_decay=weight_decay)

def build_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
    global_step: int,
):
    """
    HF schedulers accept last_epoch. For correct resume:
      last_epoch = global_step - 1
    so the NEXT scheduler.step() will move from global_step to global_step+1.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    ensure_initial_lr(optimizer)
    last_epoch = int(global_step) - 1
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        last_epoch=last_epoch,
    )

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

# -----------------------------
# CHECKPOINT I/O (MATCH KEYS!)
# -----------------------------
def save_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    global_step: int = 0,
    best_score: Optional[float] = None,
    bad_epochs: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    ckpt = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics or {},
        "history": history or [],
        "best_score": best_score,
        "bad_epochs": bad_epochs,
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint_raw(path: str, device: str = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)
def print_trainable(model):
    tot = 0
    tr = 0
    for n,p in model.named_parameters():
        tot += p.numel()
        if p.requires_grad:
            tr += p.numel()
    print(f"Total params: {tot:,} | Trainable: {tr:,} ({100*tr/tot:.2f}%)")
# -----------------------------
# TRAIN
# -----------------------------
def train(
    model,
    train_loader,
    val_loader,
    label_mem: torch.Tensor,
    concept2idx: Dict[str, int],
    device: torch.device,
    out_dir: str,
    epochs: int = 10,
    optimizer_name: str = "adamw",
    lr: float = 5e-5,
    weight_decay: float = 0.02,
    warmup_ratio: float = 0.1,
    use_scheduler: bool = True,
    grad_clip: float = 1.0,
    max_steps_gen: int = 32,
    early_stopping_patience: int = 3,
    metric_to_track: str = "micro_f1",
    resume_from: Optional[str] = None,
    runBeamSearch: bool = False,

    # -------- FLAGS you asked for --------
    freeze_enabled: bool = False,                 # if False -> encoder always trainable
    freeze_epoch: int = DEFAULT_FREEZE_EPOCH,     # freeze beginning at this epoch if enabled
    resume_load_optimizer: bool = True,           # load optimizer state if compatible
    resume_load_scheduler: bool = True,           # load scheduler state if compatible
    # When stage changes (unfreeze->freeze or freeze->unfreeze), we must rebuild optimizer.
    # For scheduler: you can either restart the schedule for the new stage (recommended) or continue global schedule.
    scheduler_on_stage_change: str = "restart",   # "restart" | "continue"
):
    """
    Supports:
      - Resume model + metadata always
      - Resume optimizer/scheduler if SAME stage and requested
      - Optional freezing encoder after some epoch (beginning of that epoch)
      - Correct LR continuation on resume using global_step + last_epoch
    """
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    idx2concept = build_idx2concept(concept2idx)
    label_universe = list(concept2idx.keys())   # e.g. ["1163","1325",...]
    print(f"Whole universe: {label_universe}")

    mlb = make_mlb_from_idx2concept(idx2concept)
    print("mlb classes:", len(mlb.classes_))
    print("example class types:", type(mlb.classes_[0]), mlb.classes_[0])
    # ---- defaults ----
    start_epoch = 1
    global_step = 0
    best_score = 1e9 if metric_to_track == "val_loss" else -1e9
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    ckpt = None

    # Compute the "global schedule" steps for the entire run (not remaining)
    steps_per_epoch = max(1, len(train_loader))
    total_steps_full = int(epochs * steps_per_epoch)
    warmup_steps_full = int(total_steps_full * warmup_ratio)

    # -----------------------------
    # 1) LOAD CHECKPOINT (MODEL + METADATA) FIRST
    # -----------------------------
    if resume_from is not None and os.path.exists(resume_from):
        ckpt = load_checkpoint_raw(resume_from, device="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        history = ckpt.get("history", []) or []
        best_score = ckpt.get("best_score", best_score)
        bad_epochs = ckpt.get("bad_epochs", bad_epochs)

        print(f"Resumed MODEL from {resume_from} | start_epoch={start_epoch} | global_step={global_step}")

    # -----------------------------
    # 2) APPLY STAGE (freeze/unfreeze) FOR start_epoch BEFORE building optimizer
    # -----------------------------
    if should_be_frozen(start_epoch, freeze_enabled=freeze_enabled, freeze_epoch=freeze_epoch):
        freeze_encoder(model)
        stage = "frozen"
        print(f"Encoder set to FROZEN at start_epoch={start_epoch} (freeze_enabled={freeze_enabled}, freeze_epoch={freeze_epoch})")
    else:
        #freeze_encoder(model)
        unfreeze_encoder_last4_only(model)
        stage = "unfrozen"
        print(f"Encoder set to TRAINABLE at start_epoch={start_epoch} (freeze_enabled={freeze_enabled}, freeze_epoch={freeze_epoch})")
    # print("XLM-R:", count_trainable("encoder"))
    # print("Decoder:", count_trainable("decoder"))
    # print("MPNet:", count_trainable("label_encoder"))
    # print("model.encoder class:", model.encoder.__class__)
    # print("encoder config model_type:", getattr(getattr(model.encoder, "config", None), "model_type", None))
    # print("encoder name_or_path:", getattr(getattr(model.encoder, "config", None), "_name_or_path", None))
    # -----------------------------
    # 3) BUILD OPTIMIZER / SCHEDULER FOR CURRENT STAGE
    # -----------------------------
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=total_steps_full,
            warmup_steps=warmup_steps_full,
            global_step=global_step,  # correct resume
        )

    # -----------------------------
    # 4) LOAD OPTIMIZER/SCHEDULER IF COMPATIBLE (SAME STAGE)
    # -----------------------------
    if ckpt is not None:
        ckpt_extra = ckpt.get("extra", {}) or {}
        ckpt_stage = ckpt_extra.get("stage", None)

        # If older checkpoints have no stage, treat as compatible ONLY if user wants
        same_stage = (ckpt_stage is None) or (ckpt_stage == stage)

        if resume_load_optimizer and same_stage and ckpt.get("optimizer_state") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                print(f"Loaded optimizer state (stage={stage})")
            except Exception as e:
                print(f"Could not load optimizer state; using rebuilt optimizer. Reason: {e}")

        else:
            print(f"Not loading optimizer state (resume_load_optimizer={resume_load_optimizer}, ckpt_stage={ckpt_stage}, current_stage={stage}).")

        if scheduler is not None and resume_load_scheduler and same_stage and ckpt.get("scheduler_state") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
                print(f"Loaded scheduler state (stage={stage})")
            except Exception as e:
                print(f"Could not load scheduler state; using rebuilt scheduler. Reason: {e}")
        else:
            if scheduler is not None:
                print(f"Not loading scheduler state (resume_load_scheduler={resume_load_scheduler}, ckpt_stage={ckpt_stage}, current_stage={stage}).")

    # -----------------------------
    # 5) MOVE TO DEVICE (AFTER LOADS)
    # -----------------------------
    model.to(device)
    label_mem = label_mem.to(device)
    optimizer_to_device(optimizer, device)

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    print_trainable(model)
    for epoch in range(start_epoch, epochs + 1):

        # Stage switch at beginning of epoch if enabled
        want_frozen = should_be_frozen(epoch, freeze_enabled=freeze_enabled, freeze_epoch=freeze_epoch)
        current_stage = stage
        next_stage = "frozen" if want_frozen else "unfrozen"

        if next_stage != current_stage:
            # apply stage change
            if next_stage == "frozen":
                print(f"Stage switch: freezing encoder at BEGIN epoch {epoch}")
                freeze_encoder(model)
            else:
                print(f"Stage switch: unfreezing encoder at BEGIN epoch {epoch}")
                unfreeze_encoder(model)

            stage = next_stage

            # MUST rebuild optimizer because param set changed
            optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
            optimizer_to_device(optimizer, device)

            # scheduler handling on stage change
            if use_scheduler:
                if scheduler_on_stage_change == "continue":
                    # keep global schedule position (based on global_step)
                    scheduler = build_scheduler(
                        optimizer=optimizer,
                        total_steps=total_steps_full,
                        warmup_steps=warmup_steps_full,
                        global_step=global_step,
                    )
                else:
                    # restart schedule from now (common choice for stage-2)
                    # (still uses full total_steps, but we "reset" to last_epoch=-1 by passing global_step=0)
                    scheduler = build_scheduler(
                        optimizer=optimizer,
                        total_steps=total_steps_full,
                        warmup_steps=warmup_steps_full,
                        global_step=0,
                    )
            else:
                scheduler = None

        # training mode
        model.train()
        if stage == "frozen":
            model.encoder.eval()

        t0 = time.time()
        running_loss = 0.0
        train_f1_sum = 0.0
        train_macro_f1_sum = 0.0
        jaccard = 0.0
        hamming = 0.0
        sub_Acc = 0.0
        card_error = 0.0
        # ---- K statistics accumulators (epoch-level) ----
        sum_pred_k = 0.0
        sum_gold_k = 0.0
        sum_gold_k_hat = 0.0
        sum_k_hat = 0.0
        num_batches = 0
        for step_in_epoch, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

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
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(get_trainable_params(model), grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item())
            if not runBeamSearch:
                pred_idxs = model.generate_greedy(
                      doc_input_ids=batch.doc_input_ids,
                      doc_attention_mask=batch.doc_attention_mask,
                      label_mem=label_mem,
                      max_new_tokens=max_steps_gen,
                      no_repeat_labels=True,
                      min_labels=model.min_labels,
                )
            else:
              pred_idxs = model.generate_beam(
                  doc_input_ids=batch.doc_input_ids,
                  doc_attention_mask=batch.doc_attention_mask,
                  label_mem=label_mem,
                  beam_size=5,
                  max_new_tokens=max_steps_gen,
                  no_repeat_labels=True,
                  min_labels=None,
                  length_penalty=0.2,
              )
            # ---- greedy metrics (debug-ish) ----

            pred_concepts = [{idx2concept[i] for i in preds} for preds in pred_idxs]
            gold_concepts = [
                set(item["concepts"] if "concepts" in item else item["labels"])
                for item in batch.raw_items
            ]
            stats = compute_k_statistics(
                pred_indices=pred_idxs,
                gold_concepts=gold_concepts,
                count_logits=out.get("count_logits"),
                max_pred_labels=model.max_pred_labels,
            )
            sum_pred_k += stats["avg_pred_k"]
            sum_gold_k += stats["avg_gold_k"]

            if stats["avg_gold_k_hat"] is not None:
                sum_gold_k_hat += stats["avg_gold_k_hat"]

            if stats["avg_k_hat"] is not None:
                sum_k_hat += stats["avg_k_hat"]

            num_batches += 1
            metrics_micro = micro_f1(pred_concepts, gold_concepts)
            metrics_macro = macro_f1_labelwise(pred_concepts, gold_concepts)
            metrics_jaccard = jaccard_score_multilabel(pred_concepts, gold_concepts,mlb)
            metrics_hamming = hamming_loss_multilabel(pred_concepts, gold_concepts,mlb)
            metrics_sub_acc = subset_accuracy(pred_concepts, gold_concepts)
            cardinality_error = label_cardinality_error(pred_concepts, gold_concepts)
            metrics = {**metrics_micro, **metrics_macro, **metrics_jaccard,**metrics_hamming, **metrics_sub_acc,**cardinality_error,**stats}

            train_f1_sum += float(metrics["micro_f1"])
            train_macro_f1_sum += float(metrics["macro_f1"])
            jaccard += float(metrics_jaccard["jaccard"])
            hamming += float(metrics_hamming["hamming_loss"])
            sub_Acc += float(metrics_sub_acc["subset_accuracy"])
            card_error += float(cardinality_error["cardinality_error"])

            if step_in_epoch == 1 or step_in_epoch % 10 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} | step {step_in_epoch}/{len(train_loader)} | global_step={global_step} | "
                    f"loss={loss.item():.4f} microF1={metrics['micro_f1']:.4f} macroF1={metrics['macro_f1']:.4f} "
                    f"jaccard={metrics_jaccard['jaccard']:.4f} hamming={metrics_hamming['hamming_loss']:.4f} "
                    f"subAcc={metrics_sub_acc['subset_accuracy']:.4f} cardinality={cardinality_error["cardinality_error"]:.4f} "
                    f"lr={cur_lr:.2e} stage={stage}"
                )

        dt = time.time() - t0
        train_loss = running_loss / max(len(train_loader), 1)
        train_micro_f1 = train_f1_sum / max(len(train_loader), 1)
        train_macro_f1 = train_macro_f1_sum / max(len(train_loader), 1)
        train_jaccard = jaccard / max(len(train_loader), 1)
        train_hamming = hamming / max(len(train_loader), 1)
        train_sub_acc = sub_Acc / max(len(train_loader), 1)
        train_card_error = card_error / max(len(train_loader), 1)
        train_avg_pred_k = sum_pred_k / max(1, num_batches)
        train_avg_gold_k = sum_gold_k / max(1, num_batches)
        train_avg_gold_k_hat = sum_gold_k_hat / max(1, num_batches)
        train_avg_k_hat = sum_k_hat / max(1, num_batches)
        metrics_val = evaluate(model, val_loader, label_mem, concept2idx, device, max_steps=max_steps_gen,runBeamSearch=runBeamSearch,mlb=mlb)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_micro_f1": float(train_micro_f1),
            "train_macro_f1": float(train_macro_f1),
            "train_jaccard": float(train_jaccard),
            "train_hamming": float(train_hamming),
            "train_subset_accuracy": float(train_sub_acc),
            "train_cardinality_error": float(train_card_error),
            "train_avg_pred_k": float(train_avg_pred_k),
            "train_avg_gold_k": float(train_avg_gold_k),
            "train_avg_gold_k_hat": float(train_avg_gold_k_hat),
            "train_avg_k_hat": float(train_avg_k_hat),

            "val_loss": float(metrics_val["val_loss"]),
            "val_micro_p": float(metrics_val["micro_p"]),
            "val_micro_r": float(metrics_val["micro_r"]),
            "val_micro_f1": float(metrics_val["micro_f1"]),
            "val_macro_p": float(metrics_val["macro_p"]),
            "val_macro_r": float(metrics_val["macro_r"]),
            "val_macro_f1": float(metrics_val["macro_f1"]),
            "val_jaccard": float(metrics_val['jaccard']),
            "val_hamming": float(metrics_val['hamming_loss']),
            "val_subset_accuracy": float(metrics_val["subset_accuracy"]),
            "val_cardinality_error": float(metrics_val["cardinality_error"]),
            "val_avg_pred_k": float(metrics_val["avg_pred_k"]),
            "val_avg_gold_k": float(metrics_val["avg_gold_k"]),
            "val_avg_gold_k_hat": float(metrics_val["avg_gold_k_hat"]),
            "val_avg_k_hat": float(metrics_val["avg_k_hat"]),
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{epochs} | train_loss={train_loss:.4f} | time={dt:.1f}s | val={metrics_val}")

        # ---- save last ----
        save_checkpoint(
            last_path,
            model, optimizer, scheduler,
            epoch=epoch,
            metrics=metrics_val,
            history=history,
            global_step=global_step,
            best_score=best_score,
            bad_epochs=bad_epochs,
            extra={
                "num_labels": getattr(model, "num_labels", None),
                "stage": stage,
                "freeze_enabled": bool(freeze_enabled),
                "freeze_epoch": int(freeze_epoch),
                "total_steps_full": int(total_steps_full),
                "warmup_steps_full": int(warmup_steps_full),
            },
        )

        current = metrics_val["micro_f1"] if metric_to_track != "val_loss" else metrics_val["val_loss"]
        improved = (current > best_score) if metric_to_track != "val_loss" else (current < best_score)

        if improved:
            best_score = current
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model, optimizer, scheduler,
                epoch=epoch,
                metrics=metrics_val,
                history=history,
                global_step=global_step,
                best_score=best_score,
                bad_epochs=bad_epochs,
                extra={
                    "num_labels": getattr(model, "num_labels", None),
                    "stage": stage,
                    "freeze_enabled": bool(freeze_enabled),
                    "freeze_epoch": int(freeze_epoch),
                    "total_steps_full": int(total_steps_full),
                    "warmup_steps_full": int(warmup_steps_full),
                },
            )
            print(f"saved best to {best_path} ({metric_to_track}={best_score:.6f})")
        else:
            bad_epochs += 1
            print(f"No improvement ({bad_epochs}/{early_stopping_patience})")

        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if bad_epochs >= early_stopping_patience:
            print("early stopping triggered")
            break

    print("done. best:", best_score, "best_path:", best_path)
