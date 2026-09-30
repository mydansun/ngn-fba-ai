"""Packed LSTM inference, preserved from nga-clean 22ec63c."""
from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)
PAD, UNK = "<PAD>", "<UNK>"
_ASCII_PRINTABLE_RE = re.compile(r"[ -~]")

def filter_ascii_printable(text: str, keep_space: bool = True) -> str:
    """
    Keep ASCII printable chars only (0x20..0x7E). Convert any whitespace to ' ' first.
    Deletion-only.
    """
    if not text:
        return ""
    text = re.sub(r"\s", " ", text)
    kept = "".join(ch for ch in text if _ASCII_PRINTABLE_RE.fullmatch(ch))
    if not keep_space:
        kept = kept.replace(" ", "")
    return kept

class CharKeepDropLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        lengths = input_ids.ne(0).sum(1).clamp_min(1).cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=input_ids.shape[1])
        x = self.drop(x)
        logits = self.out(x).squeeze(-1)
        return logits

def load_vocab(vocab_path: Path) -> Dict[str, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.json not found: {vocab_path}")
    obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = {str(k): int(v) for k, v in obj.items()}
    if PAD not in vocab or UNK not in vocab:
        logger.warning("vocab.json missing PAD/UNK tokens; inference may break for padding/unknown chars.")
    return vocab

@dataclass
class InferenceConfig:
    emb_dim: int
    hidden: int
    layers: int
    dropout: float
    thr: float
    max_len: int

def extract_hparams_from_ckpt(ckpt: Dict[str, Any]) -> InferenceConfig:
    hp = ckpt.get("hyper_parameters") or {}
    # these keys come from LightningModule.save_hyperparameters()
    return InferenceConfig(
        emb_dim=int(hp["emb_dim"]),
        hidden=int(hp["hidden"]),
        layers=int(hp["layers"]),
        dropout=float(hp["dropout"]),
        thr=float(hp.get("thr", 0.5)),
        # max_len is not necessarily stored; default to 256 if missing
        max_len=int(hp.get("max_len", 256)),
    )

def load_model_from_ckpt(ckpt_path: Path, vocab: Dict[str, int], device: torch.device) -> Tuple[
    CharKeepDropLSTM, InferenceConfig]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Invalid Lightning .ckpt: missing state_dict")

    cfg = extract_hparams_from_ckpt(ckpt)
    sd = ckpt["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("Invalid Lightning .ckpt: state_dict is not dict")

    model = CharKeepDropLSTM(
        vocab_size=len(vocab),
        emb_dim=cfg.emb_dim,
        hidden=cfg.hidden,
        num_layers=cfg.layers,
        dropout=cfg.dropout,
    )

    # Strip "model." prefix because LightningModule stores params as "model.xxx"
    fixed_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        kk = k[len("model."):] if k.startswith("model.") else k
        fixed_sd[kk] = v

    missing, unexpected = model.load_state_dict(fixed_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.to(device)
    model.eval()
    return model, cfg

@dataclass
class CleanItemMeta:
    input_len: int
    used_len: int
    kept_len: int
    keep_ratio: float
    confidence_mean: float
    confidence_min: float

class Cleaner:
    def __init__(self, model: CharKeepDropLSTM, vocab: Dict[str, int], cfg: InferenceConfig, device: torch.device):
        self.model = model
        self.vocab = vocab
        self.cfg = cfg
        self.device = device
        self.pad_id = vocab.get(PAD, 0)
        self.unk_id = vocab.get(UNK, 1)

    @torch.no_grad()
    def clean_batch_with_meta(self, ss: List[str], thr: float, max_len: int, max_batch: int):
        if not ss:
            raise ValueError("texts must be a non-empty list")
        if len(ss) > max_batch:
            raise ValueError(f"Batch too large: {len(ss)} > {max_batch}")

        normed: List[str] = []
        input_lens: List[int] = []
        used_lens: List[int] = []

        for s in ss:
            s_in = s or ""
            input_lens.append(len(s_in))
            s0 = filter_ascii_printable(s_in, keep_space=True)
            s0 = s0[:max_len] if max_len > 0 else s0
            used_lens.append(len(s0))
            normed.append(s0)

        max_t = max((len(s) for s in normed), default=0)
        if max_t == 0:
            cleaned = [""] * len(ss)
            items = [
                CleanItemMeta(
                    input_len=input_lens[i],
                    used_len=used_lens[i],
                    kept_len=0,
                    keep_ratio=0.0,
                    confidence_mean=0.0,
                    confidence_min=0.0,
                )
                for i in range(len(ss))
            ]
            return cleaned, items

        input_ids = torch.full((len(ss), max_t), self.pad_id, dtype=torch.long, device=self.device)
        for i, s0 in enumerate(normed):
            if not s0:
                continue
            ids = [self.vocab.get(ch, self.unk_id) for ch in s0]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        logits = self.model(input_ids)  # (B,T)
        probs = torch.sigmoid(logits).detach().cpu()  # (B,T)
        keep = probs >= thr

        cleaned: List[str] = []
        items: List[CleanItemMeta] = []

        for i, s0 in enumerate(normed):
            L = len(s0)
            if L == 0:
                cleaned.append("")
                items.append(
                    CleanItemMeta(
                        input_len=input_lens[i],
                        used_len=used_lens[i],
                        kept_len=0,
                        keep_ratio=0.0,
                        confidence_mean=0.0,
                        confidence_min=0.0,
                    )
                )
                continue

            keep_i = keep[i, :L]
            probs_i = probs[i, :L]

            out = "".join(ch for ch, k in zip(s0, keep_i.tolist()) if k)
            cleaned.append(out)

            kept_len = int(keep_i.sum().item())
            if kept_len > 0:
                kept_probs = probs_i[keep_i]
                conf_mean = float(kept_probs.mean().item())
                conf_min = float(kept_probs.min().item())
            else:
                conf_mean = 0.0
                conf_min = 0.0

            items.append(
                CleanItemMeta(
                    input_len=int(input_lens[i]),
                    used_len=int(used_lens[i]),
                    kept_len=kept_len,
                    keep_ratio=float(kept_len / max(1, L)),
                    confidence_mean=conf_mean,
                    confidence_min=conf_min,
                )
            )

        return cleaned, items
