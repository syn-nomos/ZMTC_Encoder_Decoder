# =========================
# MPNet + T5Stack Pointer with EOS + Count-Bias (stable greedy stopping)
# =========================
# - Frozen MPNet encoder (supports chunked token memory with dedup)
# - Trainable enc_proj -> d_model
# - T5Stack decoder (config, not pretrained) takes previous label embeddings as inputs_embeds
# - Pointer logits via dot-product with label_mem (label description embeddings)
# - EOS token is an extra class (index = L)
# - Count head predicts expected K and is used as a SOFT BIAS to EOS during greedy generation
#   (NOT hard-K). This keeps "learn EOS stopping" logic but stabilizes training/inference.
#
# Requirements:
#   label_mem: torch.Tensor [L, D] precomputed label description embeddings (same d_model)
#   dec_prev_indices & tgt_label_indices: indices in [0..L-1] and EOS=L, BOS=-1, PAD=-100


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, List
from transformers import AutoModel, T5Config
from transformers.models.t5.modeling_t5 import T5Stack


class EncT5PointerEOSCountBias(nn.Module):
    """
    Stable generative set prediction:
      - Decoder predicts labels step-by-step and stops via EOS (no beam search).
      - Count head provides a soft prior to bias EOS logit as steps progress.

    Training:
      - Pointer CE on targets (labels + EOS)
      - Count CE on K (number of labels)

    Inference:
      - Greedy decoding with EOS; EOS logit gets bias based on predicted K_hat:
          if step < K_hat:  eos_logit -= beta
          else:             eos_logit += beta
    """

    BOS_INDEX = -1
    PAD_INDEX = -100

    def __init__(
        self,
        mpnet_name: str = None,
        num_labels: int = 0,
        d_model: int = 768,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_mpnet: bool = False,
        mpnet_ft_mode: str = "last",
        # projection
        use_enc_proj: bool = False,
        freeze_enc_proj: bool = False,
        encoder_obj_proj = None,
        # label mem handling
        normalize_label_mem: bool = True,
        train_label_in_proj: bool = True,

        # chunk token memory
        encoder_memory_mode: str = "token_dedup",  # "token_flat" | "token_dedup"
        chunk_len: int = 512,
        chunk_stride: int = 256,

        # training/inference behavior
        forbid_repeats: bool = True,
        min_labels: int = 1,

        # count head
        max_pred_labels: int = 32,
        len_loss_weight: float = 1.0,

        # EOS bias from count prior
        eos_bias_beta: float = 2.0,           # strength of EOS bias
        eos_bias_temperature: float = 1.0,    # optional temperature for count distribution (usually 1)
    ):
        super().__init__()
        assert encoder_memory_mode in {"token_flat", "token_dedup"}

        self.num_labels = int(num_labels)  # L
        self.d_model = int(d_model)
        self.normalize_label_mem = bool(normalize_label_mem)
        self.forbid_repeats = bool(forbid_repeats)
        self.min_labels = int(min_labels)

        self.encoder_memory_mode = encoder_memory_mode
        self.chunk_len = int(chunk_len)
        self.chunk_stride = int(chunk_stride)

        self.max_pred_labels = int(max_pred_labels)
        self.len_loss_weight = float(len_loss_weight)

        self.eos_bias_beta = float(eos_bias_beta)
        self.eos_bias_temperature = float(eos_bias_temperature)

        # logit scale for pointer (start sharper than 1.0 helps training)
        # exp(2.3) ~ 10
        self.logit_scale = nn.Parameter(torch.tensor(2.3))

        # encoder
        if encoder_obj_proj is not None:
            self.encoder = encoder_obj_proj
        else:
            self.encoder = AutoModel.from_pretrained(mpnet_name)
        mpnet_hidden = self.encoder.config.hidden_size if hasattr(self.encoder.config, "hidden_size") else 768
        if freeze_mpnet:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        else:
            self.mpnet_ft_mode = mpnet_ft_mode
            #self._set_mpnet_finetune_mode(mpnet_ft_mode, unfreeze_layer_norms=True)
            self._set_encoder_finetune_mode(mpnet_ft_mode, unfreeze_layer_norms=True)
        # decoder (config)
        cfg = T5Config(
            vocab_size=1,
            d_model=self.d_model,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=False,
        )
        self.decoder = T5Stack(cfg) #, embed_tokens=None

        # enc projection to d_model
        if use_enc_proj:
            self.enc_proj = nn.Linear(mpnet_hidden, self.d_model)
            self.enc_ln = nn.LayerNorm(self.d_model)
            if freeze_enc_proj:
                for p in self.enc_proj.parameters():
                    p.requires_grad = False
                for p in self.enc_ln.parameters():
                    p.requires_grad = False
        else:
            self.enc_proj = None
            self.enc_ln = None
            if mpnet_hidden != self.d_model:
                raise ValueError("mpnet_hidden != d_model and use_enc_proj=False")

        # decoder input embeddings: BOS/EOS vectors
        self.bos_in = nn.Parameter(torch.randn(self.d_model) * 0.02)
        self.eos_in = nn.Parameter(torch.randn(self.d_model) * 0.02)

        # optional adapter for feeding label embeddings into decoder inputs
        self.label_in_proj = nn.Linear(self.d_model, self.d_model, bias=False) if train_label_in_proj else nn.Identity()

        # pointer head
        self.q_ln = nn.LayerNorm(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.eos_head = nn.Linear(self.d_model, 1)

        # count head over doc vector
        self.count_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.max_pred_labels + 1),  # K in [0..maxK]
        )

    # ---------- utils ----------
    def _apply_enc_proj(self, x: torch.Tensor) -> torch.Tensor:
        if self.enc_proj is None:
            return x
        return self.enc_ln(self.enc_proj(x))
    def _get_encoder_layers(self) -> nn.ModuleList:
      """
      Return the transformer block list (ModuleList) for many HF encoders:
      - MPNetModel:        encoder.layer
      - DistilBertModel:   transformer.layer
      - Bert/Roberta:      encoder.layer
      - DebertaV2/V3:      encoder.layer
      - Some models:       layers / encoder.layers
      """
      candidates = [
          # DistilBERT
          ("transformer", "layer"),
          ("distilbert", "transformer", "layer"),  # sometimes wrapped (rare)

          # BERT/RoBERTa/MPNet/DeBERTa style
          ("encoder", "layer"),
          ("encoder", "layers"),

          # fallback
          ("layers",),
      ]

      for path in candidates:
          obj = self.encoder
          ok = True
          for attr in path:
              if not hasattr(obj, attr):
                  ok = False
                  break
              obj = getattr(obj, attr)

          if ok and isinstance(obj, nn.ModuleList):
              return obj

      raise AttributeError(
          f"Could not locate transformer layers for encoder class: {self.encoder.__class__.__name__}. "
          "Tried common paths like encoder.layer and transformer.layer."
      )
    def _set_encoder_finetune_mode(self, mode: str, unfreeze_layer_norms: bool = True):
      """
      mode:
        - "frozen": freeze everything
        - "last":   unfreeze only last transformer block (+ optionally LN)
        - "all":    unfreeze all encoder params
      """
      mode = mode.lower().strip()
      assert mode in {"frozen", "last", "all"}

      # freeze all by default
      for p in self.encoder.parameters():
          p.requires_grad = False

      if mode == "all":
          for p in self.encoder.parameters():
              p.requires_grad = True

      elif mode == "last":
          layers = self._get_encoder_layers()
          print("Encoder class:", self.encoder.__class__.__name__)
          print("Num encoder blocks found:", len(layers))
          print("Last block type:", type(layers[-1]))
          for layer in layers[-4:]:
            for p in layer.parameters():
              p.requires_grad = True

          if unfreeze_layer_norms:
              for m in self.encoder.modules():
                  if isinstance(m, torch.nn.LayerNorm):
                      for p in m.parameters():
                          p.requires_grad = True

      any_trainable = any(p.requires_grad for p in self.encoder.parameters())
      self.encoder.train() if any_trainable else self.encoder.eval()
      return any_trainable
    def _masked_mean_pool(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mf = m.unsqueeze(-1).float()
        return (h * mf).sum(1) / mf.sum(1).clamp(min=1e-6)

    def _encode_chunks_token_memory(self, doc_input_ids: torch.Tensor, doc_attention_mask: torch.Tensor):
        # doc_input_ids: [B,C,S]
        B, C, S = doc_input_ids.shape
        flat_ids = doc_input_ids.reshape(B * C, S)
        flat_msk = doc_attention_mask.reshape(B * C, S)

        out = self.encoder(input_ids=flat_ids, attention_mask=flat_msk, return_dict=True)
        tok = out.last_hidden_state  # [B*C,S,H]
        H = tok.size(-1)
        tok = tok.view(B, C, S, H)
        msk = doc_attention_mask

        if self.encoder_memory_mode == "token_flat":
            enc_h = tok.reshape(B, C * S, H)
            enc_m = msk.reshape(B, C * S)
        else:
            keep = int(self.chunk_len - self.chunk_stride)
            if keep <= 0 or keep > S:
                raise ValueError(f"Bad keep={keep} from chunk_len={self.chunk_len}, stride={self.chunk_stride}, S={S}")
            pieces, masks = [], []
            for c in range(C):
                if c == 0:
                    pieces.append(tok[:, c, :, :])
                    masks.append(msk[:, c, :])
                else:
                    pieces.append(tok[:, c, -keep:, :])
                    masks.append(msk[:, c, -keep:])
            enc_h = torch.cat(pieces, dim=1)
            enc_m = torch.cat(masks, dim=1)

        enc_h = self._apply_enc_proj(enc_h)  # [B,S_mem,D]
        enc_m = enc_m.long()
        return enc_h, enc_m

    def encode(self, doc_input_ids, doc_attention_mask):
        if doc_input_ids.dim() == 2:
            out = self.encoder(doc_input_ids, attention_mask=doc_attention_mask, return_dict=True)
            h = self._apply_enc_proj(out.last_hidden_state)
            return h, doc_attention_mask.long()
        if doc_input_ids.dim() == 3:
            return self._encode_chunks_token_memory(doc_input_ids, doc_attention_mask)
        raise ValueError("doc_input_ids must be 2D or 3D")

    def _build_decoder_inputs_embeds(self, dec_prev_indices: torch.Tensor, label_mem_in: torch.Tensor):
        """
        dec_prev_indices: [B,T] values:
          BOS=-1, labels 0..L-1, EOS=L, PAD=-100
        label_mem_in: [L,D]
        """
        B, T = dec_prev_indices.shape
        D = label_mem_in.size(-1)
        device = dec_prev_indices.device

        mem = self.label_in_proj(label_mem_in)  # [L,D]
        out = torch.zeros((B, T, D), device=device, dtype=mem.dtype)

        lab_mask = (dec_prev_indices >= 0) & (dec_prev_indices < self.num_labels)
        if lab_mask.any():
            idx = dec_prev_indices[lab_mask].long()
            out[lab_mask] = mem.index_select(0, idx)

        bos_mask = (dec_prev_indices == self.BOS_INDEX)
        eos_mask = (dec_prev_indices == self.num_labels)
        pad_mask = (dec_prev_indices == self.PAD_INDEX)

        if bos_mask.any():
            out[bos_mask] = self.bos_in.to(device=device, dtype=out.dtype)
        if eos_mask.any():
            out[eos_mask] = self.eos_in.to(device=device, dtype=out.dtype)
        if pad_mask.any():
            out[pad_mask] = 0.0
        return out

    # ---------- forward ----------
    def forward(
        self,
        doc_input_ids,
        doc_attention_mask,
        dec_prev_indices,
        dec_attention_mask,
        label_mem,               # [L,D]
        tgt_label_indices=None,  # [B,T] values 0..L-1 or EOS=L
        tgt_pad_mask=None,
        gold_k: Optional[torch.Tensor] = None,  # [B] optional
    ):
        enc_h, enc_m = self.encode(doc_input_ids, doc_attention_mask)  # doc token memory

        # pointer label memory: normalize for dot-product if desired
        label_mem_in = label_mem
        label_mem_ptr = label_mem
        if self.normalize_label_mem:
            label_mem_ptr = F.normalize(label_mem_ptr, dim=-1)

        if dec_attention_mask is None:
            dec_attention_mask = (dec_prev_indices != self.PAD_INDEX).long()

        dec_inputs = self._build_decoder_inputs_embeds(dec_prev_indices, label_mem_in)

        dec_out = self.decoder(
            inputs_embeds=dec_inputs,
            attention_mask=dec_attention_mask,
            encoder_hidden_states=enc_h,
            encoder_attention_mask=enc_m,
            use_cache=False,
            return_dict=True,
        )

        dec_h = dec_out.last_hidden_state  # [B,T,D]
        q = self.q_ln(dec_h)
        q = self.out_proj(q)
        q = F.normalize(q, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)

        label_logits = torch.matmul(q, label_mem_ptr.t()) * scale   # [B,T,L]
        eos_logits = self.eos_head(q).squeeze(-1) * scale           # [B,T]
        pointer_logits = torch.cat([label_logits, eos_logits.unsqueeze(-1)], dim=-1)  # [B,T,L+1]

        # count head (doc-level)
        doc_vec = self._masked_mean_pool(enc_h, enc_m)              # [B,D]
        count_logits = self.count_head(doc_vec)                     # [B,maxK+1]

        ret = {"pointer_logits": pointer_logits, "count_logits": count_logits, "loss": None}

        if tgt_label_indices is not None:
            B, T, Lp1 = pointer_logits.shape
            L = Lp1 - 1
            if tgt_pad_mask is None:
                tgt_pad_mask = (dec_attention_mask > 0).float()

            targets = tgt_label_indices.masked_fill(tgt_pad_mask == 0, -100)
            ce = F.cross_entropy(pointer_logits.view(B * T, L + 1), targets.view(B * T), ignore_index=-100)

            if gold_k is None:
                with torch.no_grad():
                    # K = number of (non-EOS) valid positions
                    gold_k = ((tgt_label_indices != L).float() * tgt_pad_mask).sum(dim=1).long()
                    gold_k = gold_k.clamp(min=0, max=self.max_pred_labels)

            len_ce = F.cross_entropy(count_logits, gold_k)

            loss = ce + self.len_loss_weight * len_ce
            ret.update({"loss": loss, "ce_loss": ce.detach(), "len_ce": len_ce.detach(), "gold_k": gold_k.detach()})

        return ret

    # ---------- generation with EOS + count-bias ----------
    @torch.no_grad()
    def generate_greedy(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        label_mem: torch.Tensor,
        max_new_tokens: int = 32,
        no_repeat_labels: bool = True,
        min_labels: Optional[int] = None,
    ) -> List[List[int]]:
        device = doc_input_ids.device
        L = label_mem.size(0)

        if min_labels is None:
            min_labels = self.min_labels

        # encode doc
        enc_h, enc_m = self.encode(doc_input_ids, doc_attention_mask)

        # pointer label memory
        label_mem_in = label_mem
        label_mem_ptr = label_mem
        if self.normalize_label_mem:
            label_mem_ptr = F.normalize(label_mem_ptr, dim=-1)

        # predict K_hat (prior) to bias EOS
        doc_vec = self._masked_mean_pool(enc_h, enc_m)
        count_logits = self.count_head(doc_vec)
        if self.eos_bias_temperature != 1.0:
            count_logits = count_logits / self.eos_bias_temperature
        k_hat = torch.argmax(count_logits, dim=-1).clamp(min=min_labels, max=min(self.max_pred_labels, max_new_tokens))  # [B]

        B = doc_input_ids.size(0)
        preds = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        used_mask = torch.zeros((B, L), dtype=torch.bool, device=device) if (no_repeat_labels and self.forbid_repeats) else None

        dec_prev = torch.full((B, 1), self.BOS_INDEX, dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            dec_attn = (dec_prev != self.PAD_INDEX).long()
            dec_inputs = self._build_decoder_inputs_embeds(dec_prev, label_mem_in)

            out = self.decoder(
                inputs_embeds=dec_inputs,
                attention_mask=dec_attn,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=enc_m,
                use_cache=False,
                return_dict=True,
            )

            q = out.last_hidden_state[:, -1, :]
            q = self.q_ln(q)
            q = self.out_proj(q)
            q = F.normalize(q, dim=-1)

            scale = self.logit_scale.exp().clamp(max=100.0)

            lab_logits = torch.matmul(q, label_mem_ptr.t()) * scale  # [B,L]
            if used_mask is not None:
                lab_logits = lab_logits.masked_fill(used_mask, float("-inf"))

            eos_logit = self.eos_head(q).squeeze(-1) * scale          # [B]

            # --- count-bias to EOS ---
            # if step < k_hat: discourage EOS; else encourage EOS
            beta = self.eos_bias_beta
            bias = torch.where(step < k_hat, -beta, +beta).to(eos_logit.dtype)
            eos_logit = eos_logit + bias * scale

            step_logits = torch.cat([lab_logits, eos_logit.unsqueeze(-1)], dim=-1)  # [B,L+1]

            # enforce min labels before EOS
            if step < min_labels:
                step_logits[:, L] = float("-inf")

            choice = torch.argmax(step_logits, dim=-1)  # 0..L (L==EOS)
            is_eos = (choice == L)

            for b in range(B):
                if finished[b]:
                    continue
                if is_eos[b]:
                    finished[b] = True
                else:
                    lbl = int(choice[b].item())
                    preds[b].append(lbl)
                    if used_mask is not None:
                        used_mask[b, lbl] = True

            # feed back previous token (label or EOS)
            next_prev = choice.clone()
            next_prev[finished] = L
            dec_prev = torch.cat([dec_prev, next_prev.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return preds
    @torch.no_grad()
    def generate_beam(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        label_mem: torch.Tensor,
        beam_size: int = 5,
        max_new_tokens: int = 32,
        no_repeat_labels: bool = True,
        min_labels: Optional[int] = None,
        length_penalty: float = 0.0,   # 0.0 = no normalization, 0.6..1.0 typical
        return_scores: bool = False,
    ):
        """
        Beam search decoding for label-sequence generation with EOS + count-bias.

        - Keeps top-K partial hypotheses (beams) by accumulated log-prob.
        - Applies the same EOS bias rule as greedy using predicted K_hat:
            if step < K_hat: eos_logit -= beta
            else:            eos_logit += beta
        - Enforces min_labels (cannot emit EOS before min_labels).
        - Optionally forbids repeated labels per hypothesis.

        
        - length_penalty is applied consistently BOTH during pruning AND in final best selection.
        """
        device = doc_input_ids.device
        L = label_mem.size(0)

        if min_labels is None:
            min_labels = self.min_labels

        beam_size = int(max(1, beam_size))
        max_new_tokens = int(max(1, max_new_tokens))

        # ---------- encode doc ----------
        enc_h, enc_m = self.encode(doc_input_ids, doc_attention_mask)

        # pointer label memory
        label_mem_in = label_mem
        label_mem_ptr = label_mem
        if self.normalize_label_mem:
            label_mem_ptr = F.normalize(label_mem_ptr, dim=-1)

        # predict K_hat (prior) to bias EOS
        doc_vec = self._masked_mean_pool(enc_h, enc_m)
        count_logits = self.count_head(doc_vec)
        if self.eos_bias_temperature != 1.0:
            count_logits = count_logits / self.eos_bias_temperature
        k_hat = torch.argmax(count_logits, dim=-1).clamp(
            min=min_labels,
            max=min(self.max_pred_labels, max_new_tokens)
        )  # [B]

        B = doc_input_ids.size(0)

        # ---------- scoring helpers (consistent scoring everywhere) ----------
        def _score_raw(c) -> float:
            return float(c["score"])

        def _score_norm(c) -> float:
            # length = number of predicted labels (not counting BOS; EOS doesn't add label)
            l = max(1, len(c["labels"]))
            return float(c["score"]) / (l ** float(length_penalty))

        def _rank_score(c) -> float:
            if length_penalty and length_penalty > 0.0:
                return _score_norm(c)
            return _score_raw(c)

        # ---------- helper: compute step logits for a batch of beams for ONE sample ----------
        def _step_logits_for_beams(
            b: int,
            dec_prev_stack: torch.Tensor,             # [Nb, T]
            used_mask_stack: Optional[torch.Tensor],  # [Nb, L] or None
            step: int,
        ) -> torch.Tensor:
            """
            Returns logits [Nb, L+1] for the next token for sample b.
            """
            Nb, T = dec_prev_stack.shape
            dec_attn = (dec_prev_stack != self.PAD_INDEX).long()
            dec_inputs = self._build_decoder_inputs_embeds(dec_prev_stack, label_mem_in)

            out = self.decoder(
                inputs_embeds=dec_inputs,
                attention_mask=dec_attn,
                encoder_hidden_states=enc_h[b:b+1].expand(Nb, -1, -1),
                encoder_attention_mask=enc_m[b:b+1].expand(Nb, -1),
                use_cache=False,
                return_dict=True,
            )

            q = out.last_hidden_state[:, -1, :]  # [Nb, D]
            q = self.q_ln(q)
            q = self.out_proj(q)
            q = F.normalize(q, dim=-1)

            scale = self.logit_scale.exp().clamp(max=100.0)

            lab_logits = torch.matmul(q, label_mem_ptr.t()) * scale  # [Nb, L]
            if used_mask_stack is not None:
                lab_logits = lab_logits.masked_fill(used_mask_stack, float("-inf"))

            eos_logit = self.eos_head(q).squeeze(-1) * scale  # [Nb]

            # count-bias to EOS (same rule as greedy)
            beta = self.eos_bias_beta
            bias = (-beta if (step < int(k_hat[b].item())) else +beta)
            eos_logit = eos_logit + eos_logit.new_full(eos_logit.shape, bias)

            step_logits = torch.cat([lab_logits, eos_logit.unsqueeze(-1)], dim=-1)  # [Nb, L+1]

            # enforce min labels before EOS
            if step < int(min_labels):
                step_logits[:, L] = float("-inf")

            return step_logits

        # ---------- beam search per sample ----------
        all_preds: List[List[int]] = []
        all_scores: List[float] = []

        for b in range(B):
            init_dec_prev = torch.tensor([self.BOS_INDEX], dtype=torch.long, device=device)
            if no_repeat_labels and self.forbid_repeats:
                init_used = torch.zeros((L,), dtype=torch.bool, device=device)
            else:
                init_used = None

            beams = [{
                "dec_prev": init_dec_prev,
                "labels": [],
                "used": init_used,
                "score": 0.0,
                "finished": False,
            }]

            for step in range(max_new_tokens):
                if all(beam["finished"] for beam in beams):
                    break

                active = [beam for beam in beams if not beam["finished"]]
                finished = [beam for beam in beams if beam["finished"]]

                Nb = len(active)
                if Nb == 0:
                    beams = finished
                    break

                # pad dec_prev to same length for stacking
                maxT = max(beam["dec_prev"].numel() for beam in active)
                dec_prev_stack = torch.full((Nb, maxT), self.PAD_INDEX, dtype=torch.long, device=device)
                for i, beam in enumerate(active):
                    t = beam["dec_prev"].numel()
                    dec_prev_stack[i, :t] = beam["dec_prev"]

                used_stack = None
                if active[0]["used"] is not None:
                    used_stack = torch.stack([beam["used"] for beam in active], dim=0)  # [Nb, L]

                # compute logits and logprobs
                logits = _step_logits_for_beams(b, dec_prev_stack, used_stack, step=step)  # [Nb, L+1]
                logprobs = F.log_softmax(logits, dim=-1)                                   # [Nb, L+1]

                # Expand: take top candidates per beam, then global top beam_size
                per_beam_topk = min(beam_size, L + 1)
                topk_logp, topk_idx = torch.topk(logprobs, k=per_beam_topk, dim=-1)       # [Nb, K]

                candidates = []
                for i, beam in enumerate(active):
                    base_score = float(beam["score"])
                    for j in range(per_beam_topk):
                        tok = int(topk_idx[i, j].item())
                        tok_lp = float(topk_logp[i, j].item())
                        new_score = base_score + tok_lp

                        new_labels = beam["labels"]
                        new_used = beam["used"]

                        if tok == L:
                            # EOS
                            new_labels2 = list(new_labels)
                            new_used2 = new_used.clone() if (new_used is not None) else None
                            new_finished = True
                            new_tok = L
                        else:
                            # label
                            new_labels2 = list(new_labels) + [tok]
                            if new_used is not None:
                                new_used2 = new_used.clone()
                                new_used2[tok] = True
                            else:
                                new_used2 = None
                            new_finished = False
                            new_tok = tok

                        # extend dec_prev by emitting tok (or EOS)
                        dec_prev2 = torch.cat(
                            [beam["dec_prev"], torch.tensor([new_tok], dtype=torch.long, device=device)],
                            dim=0
                        )

                        candidates.append({
                            "dec_prev": dec_prev2,
                            "labels": new_labels2,
                            "used": new_used2,
                            "score": new_score,
                            "finished": new_finished,
                        })

                # keep also previously finished beams (they can survive competition)
                candidates.extend(finished)

                # prune with consistent scoring
                candidates.sort(key=_rank_score, reverse=True)
                beams = candidates[:beam_size]

            # pick best: prefer finished beams; if none finished, take best ongoing
            finished_beams = [beam for beam in beams if beam["finished"]]
            if len(finished_beams) > 0:
                best = max(finished_beams, key=_rank_score)
            else:
                best = max(beams, key=_rank_score) 

            all_preds.append(best["labels"])

            # score to return: by default return the SAME score used for selection
            # (if you want raw always, change to float(best["score"]))
            all_scores.append(float(_rank_score(best)))

        if return_scores:
            return all_preds, all_scores
        return all_preds

