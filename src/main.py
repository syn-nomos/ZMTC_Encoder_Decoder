from transformers import AutoTokenizer,AutoModel
import random
from utils import (
    load_oflan_tf_into_encoder,
    load_label_descriptions,
    scan_label_count_stats,
    precompute_label_memory,
    print_trainable_parameter_summary,
    print_decoder_blocks_trainability,
)
import os
import argparse
import json
import torch
import torch.nn.functional as F
from model import EncT5PointerEOSCountBias
from dataset import JsonlDataset, EurlexPointerCollator
from torch.utils.data import DataLoader
from train import train
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

def load_config(conf_path: str) -> dict:
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load configuration."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the config JSON file.",
    )
    parser.add_argument(
        "--load-oflan",
        action="store_true", default=False,
        help="Whether to load OFLAN weights into the encoder.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    configuration = load_config(args.path)
    if args.load_oflan:
        OFLAN_PATH =  configuration.get("oflan_path")
        encoder = AutoModel.from_pretrained("distilbert-base-uncased") #oflan uses distilbert
        w_before = encoder.state_dict()["transformer.layer.0.attention.q_lin.weight"].clone()
        missing, unexpected = load_oflan_tf_into_encoder(encoder, OFLAN_PATH, strict=False)
        print("missing:", missing[:20])
        print("unexpected:", unexpected[:20])
        w_after = encoder.state_dict()["transformer.layer.0.attention.q_lin.weight"]
        diff = (w_after - w_before).abs().mean().item()
        print("mean abs diff:", diff)
    labels_path = configuration.get("elds_path")
    train_path  = configuration.get("train_jsonl_path")
    dev_path    = configuration.get("dev_jsonl_path")
    tst_path    = configuration.get("test_jsonl_path")
    sorted_concepts, concept2idx, idx2concept, concept2desc = load_label_descriptions(
        labels_jsonl_path=labels_path,
        only_used_labels=True,
        dataset_jsonl_paths=[train_path, dev_path, tst_path],
    )
    num_labels = len(sorted_concepts)
    print("num_labels:", num_labels)
    print("idx2concept:", len(idx2concept.keys()))
    
    mx_train, avg_train = scan_label_count_stats(train_path)
    mx_dev, avg_dev = scan_label_count_stats(dev_path)
    print("train labels: max", mx_train, "avg", avg_train)
    print("dev labels:   max", mx_dev, "avg", avg_dev)
    max_labels_cap = max(mx_train, mx_dev)
    max_new_tokens = max_labels_cap + 1  # +1 for EOS

    label_encoder = configuration.get("label_encoder")
    doc_encoder = configuration.get("document_encoder")
    print("Loading document Tokenizer...")
    doc_tok = AutoTokenizer.from_pretrained(doc_encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_path = configuration.get("output_path")

    model = EncT5PointerEOSCountBias(
        mpnet_name=doc_encoder if not bool(args.load_oflan) else None,
        num_labels=num_labels,
        d_model=configuration.get("decoder_hidden_size"),
        num_layers=configuration.get("decoder_layers"),
        num_heads=configuration.get("decoder_attn_heads"),
        encoder_obj_proj=None if not args.load_oflan else encoder,
        freeze_mpnet=bool(configuration.get("freeze_doc_encoder")),
        use_enc_proj=False,

        encoder_memory_mode="token_dedup",
        chunk_len=int(configuration.get("doc_chunk_size")),
        chunk_stride=int(configuration.get("stride_window")),

        train_label_in_proj=True,
        normalize_label_mem=True,

        forbid_repeats=True,
        min_labels=1,#2

        max_pred_labels=32, #32 for EURLEX and 75 for MIMIC
        len_loss_weight= float(configuration.get("lamda")), #1.0 for EURLEX and 1.2 for MIMIC

        eos_bias_beta=0.5
    )
    model.to(device)

    label_encoder = AutoModel.from_pretrained(label_encoder).to(device)
    label_tokenizer = AutoTokenizer.from_pretrained(label_encoder)
    for p in label_encoder.parameters():
        p.requires_grad = False
    label_encoder.eval()

    label_texts = [concept2desc[c] for c in sorted_concepts]

    print("Precomputing label memory...")
    label_mem = precompute_label_memory(
        labels_text=label_texts,
        tokenizer=label_tokenizer,
        encoder_model=label_encoder,
        device=device,
        pooling="mean",#"mean",
        normalize=True,
        batch_size=128,
        #max_len=512,

    )
    label_mem.requires_grad = False
    label_mem = F.normalize(label_mem, dim=-1)
    print_trainable_parameter_summary(model)
    print_decoder_blocks_trainability(model)
    print("label_mem:", label_mem.shape, label_mem.dtype, label_mem.device)

    train_ds = JsonlDataset(train_path)
    dev_ds   = JsonlDataset(dev_path)
    test_ds  = JsonlDataset(tst_path)
    collator = EurlexPointerCollator(
        doc_tok,
        concept2idx,
        num_labels=num_labels,
        chunk_docs=True,
        chunk_len=int(configuration.get("doc_chunk_size")),
        chunk_stride=int(configuration.get("stride_window")),
        max_chunks=3,
        permute_labels=True,
    )
    BATCH_SIZE = int(configuration.get("batch_size",0))
    LR = float(configuration.get("lr",3e-5))
    EPOCHS = int(configuration.get("epochs",0))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_mem=label_mem,
        concept2idx=concept2idx,
        device=device,
        out_dir=configuration.get("output_path"),

        epochs=EPOCHS,

        optimizer_name="adamw",   # or "adafactor"
        lr=LR,#1e-4,
        weight_decay=0.04,
        warmup_ratio=0.15,
        use_scheduler=True,


        freeze_enabled=False,
        freeze_epoch=30,
        runBeamSearch=False,

        resume_from=None,
        resume_load_optimizer=True,
        resume_load_scheduler=True,
        early_stopping_patience=4,
        
        scheduler_on_stage_change="continue",
    )

    print("Evaluation on Test Set ...")
    best_ckpt_path = os.path.join(resume_path, "best.pt")
    if os.path.exists(best_ckpt_path):
        from eval import evaluate
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.to(device)
        model.eval()
        print("\n================= TEST EVAL (BEST CHECKPOINT) =================")
        test_metrics_best = evaluate(model, test_loader, label_mem, concept2idx, device, max_steps=32)
        print("test(best):", test_metrics_best)
    else:
        print("best.pt not found, skipped test eval")

if __name__ == "__main__":
    main()


#python train_main.py --path config.json --load-oflan
#python train_main.py --path config.json