from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
import json
from torch.utils.data import Dataset

class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class BatchTensors:
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    dec_prev_indices: torch.Tensor
    dec_attention_mask: torch.Tensor
    tgt_indices: torch.Tensor
    tgt_pad_mask: torch.Tensor
    raw_items: Optional[List[Dict[str, Any]]] = None

    def to(self, device: torch.device):
        return BatchTensors(
            doc_input_ids=self.doc_input_ids.to(device),
            doc_attention_mask=self.doc_attention_mask.to(device),
            dec_prev_indices=self.dec_prev_indices.to(device),
            dec_attention_mask=self.dec_attention_mask.to(device),
            tgt_indices=self.tgt_indices.to(device),
            tgt_pad_mask=self.tgt_pad_mask.to(device),
            raw_items=self.raw_items,
        )


class EurlexPointerCollator:
    """
    Produces chunked docs [B,C,S] and target label sequences (permuted) + EOS.
    """

    BOS_PREV = -1
    PAD_PREV = -100

    def __init__(
        self,
        tokenizer,
        concept2idx: Dict[str, int],
        num_labels: int,
        max_doc_len: int = 5000,   # used only if chunk_docs=False
        chunk_docs: bool = True,
        chunk_len: int = 512,
        chunk_stride: int = 256,
        max_chunks: int = 8,
        permute_labels: bool = True,
        enforce_min_1_label: bool = True,
    ):
        self.tok = tokenizer
        self.concept2idx = concept2idx
        self.num_labels = int(num_labels)
        self.max_doc_len = int(max_doc_len)

        self.chunk_docs = bool(chunk_docs)
        self.chunk_len = int(chunk_len)
        self.chunk_stride = int(chunk_stride)
        self.max_chunks = int(max_chunks)

        self.permute_labels = bool(permute_labels)
        self.enforce_min_1_label = bool(enforce_min_1_label)

        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _chunk_ids(self, ids: List[int], msk: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        real_len = int(sum(msk))
        ids = ids[:real_len]
        msk = msk[:real_len]

        chunks_ids, chunks_msk = [], []
        start = 0
        while start < real_len and len(chunks_ids) < self.max_chunks:
            end = min(start + self.chunk_len, real_len)
            ci = ids[start:end]
            cm = msk[start:end]

            pad_n = self.chunk_len - len(ci)
            if pad_n > 0:
                ci = ci + [self.pad_id] * pad_n
                cm = cm + [0] * pad_n

            chunks_ids.append(ci)
            chunks_msk.append(cm)

            if end == real_len:
                break
            start += self.chunk_stride

        if len(chunks_ids) == 0:
            chunks_ids = [[self.pad_id] * self.chunk_len]
            chunks_msk = [[0] * self.chunk_len]

        # pad chunk dimension to max_chunks
        while len(chunks_ids) < self.max_chunks:
            chunks_ids.append([self.pad_id] * self.chunk_len)
            chunks_msk.append([0] * self.chunk_len)

        return torch.tensor(chunks_ids, dtype=torch.long), torch.tensor(chunks_msk, dtype=torch.long)

    def __call__(self, items: List[Dict[str, Any]]) -> BatchTensors:
        # filter empty labels
        if self.enforce_min_1_label:
            kept = []
            for it in items:
                cs = []
                if "concepts" in it.keys():
                  cs = [str(c) for c in it.get("concepts", [])]
                elif "labels" in it.keys():
                  cs = [str(c) for c in it.get("labels", [])]
                idxs = [self.concept2idx[c] for c in cs if c in self.concept2idx]
                if len(set(idxs)) > 0:
                    kept.append(it)
            if kept:
                items = kept

        texts = [build_document_text(it) for it in items]  # you already have this helper

        if not self.chunk_docs:
            enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_doc_len, return_tensors="pt")
            doc_input_ids = enc["input_ids"]           # [B,S]
            doc_attention_mask = enc["attention_mask"] # [B,S]
        else:
            enc = self.tok(texts, padding=False, truncation=False, return_attention_mask=True)
            all_ids = enc["input_ids"]
            all_msk = enc["attention_mask"]

            ids_chunks, msk_chunks = [], []
            for ids, msk in zip(all_ids, all_msk):
                ids_t, msk_t = self._chunk_ids(ids, msk)
                ids_chunks.append(ids_t)
                msk_chunks.append(msk_t)

            doc_input_ids = torch.stack(ids_chunks, dim=0)         # [B,C,S]
            doc_attention_mask = torch.stack(msk_chunks, dim=0)    # [B,C,S]

        # build target sequences
        tgt_seqs: List[List[int]] = []
        for it in items:
            cs = []
            if "concepts" in it.keys():
               cs = [str(c) for c in it.get("concepts", [])]
            elif "labels" in it.keys():
               cs = [str(c) for c in it.get("labels", [])]
            idxs = [self.concept2idx[c] for c in cs if c in self.concept2idx]
            idxs = sorted(set(idxs))

            if self.permute_labels and len(idxs) > 1:
                perm = torch.randperm(len(idxs)).tolist()
                idxs = [idxs[i] for i in perm]

            tgt_seqs.append(idxs + [self.num_labels])  # EOS=L

        B = len(items)
        max_t = max(len(s) for s in tgt_seqs)

        tgt_indices = torch.full((B, max_t), 0, dtype=torch.long)
        tgt_pad_mask = torch.zeros((B, max_t), dtype=torch.float32)

        for i, s in enumerate(tgt_seqs):
            t = len(s)
            tgt_indices[i, :t] = torch.tensor(s, dtype=torch.long)
            tgt_pad_mask[i, :t] = 1.0

        dec_prev_indices = torch.full((B, max_t), self.PAD_PREV, dtype=torch.long)
        dec_prev_indices[:, 0] = self.BOS_PREV
        for i, s in enumerate(tgt_seqs):
            t = len(s)
            if t > 1:
                dec_prev_indices[i, 1:t] = torch.tensor(s[:-1], dtype=torch.long)

        dec_attention_mask = (tgt_pad_mask > 0).long()

        return BatchTensors(
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            dec_prev_indices=dec_prev_indices,
            dec_attention_mask=dec_attention_mask,
            tgt_indices=tgt_indices,
            tgt_pad_mask=tgt_pad_mask,
            raw_items=items,
        )