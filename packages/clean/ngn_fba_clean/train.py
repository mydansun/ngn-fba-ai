#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("keepdrop")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


# -----------------------------
# Utils
# -----------------------------
PAD = "<PAD>"
UNK = "<UNK>"

_ASCII_PRINTABLE_RE = re.compile(r"[ -~]")  # 0x20..0x7E


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_ascii_printable(text: str, keep_space: bool = True) -> str:
    """
    Keep ASCII printable chars only (0x20..0x7E). Convert any whitespace to ' ' first.
    Deletion-only (safe).
    """
    if not text:
        return ""
    text = re.sub(r"\s", " ", text)
    kept = "".join(ch for ch in text if _ASCII_PRINTABLE_RE.fullmatch(ch))
    if not keep_space:
        kept = kept.replace(" ", "")
    return kept


def maybe_casefold(s: str, casefold: bool) -> str:
    return s.upper() if casefold else s


def greedy_subseq_keep_mask(raw: str, clean: str, casefold: bool) -> Tuple[bool, List[int]]:
    r = maybe_casefold(raw, casefold)
    c = maybe_casefold(clean, casefold)

    mask = [0] * len(raw)
    j = 0
    for i in range(len(r)):
        if j < len(c) and r[i] == c[j]:
            mask[i] = 1
            j += 1
    return (j == len(c)), mask


def lcs_keep_mask(raw: str, clean: str, casefold: bool) -> List[int]:
    r = maybe_casefold(raw, casefold)
    c = maybe_casefold(clean, casefold)
    n, m = len(r), len(c)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ri = r[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            if ri == c[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] >= prev[j] else prev[j]

    mask = [0] * n
    i, j = n, m
    while i > 0 and j > 0:
        if r[i - 1] == c[j - 1]:
            mask[i - 1] = 1
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
    return mask


def load_rows_from_csv(
    path: str,
    raw_col: str,
    clean_col: str,
    file_col: Optional[str],
    max_len: int,
) -> List[Dict[str, str]]:
    """
    Load *all* rows and keep all original columns, plus normalized raw/clean.
    We keep original row fields so export can round-trip.
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        if raw_col not in cols or clean_col not in cols:
            raise SystemExit(f"CSV {path} must contain columns {raw_col} and {clean_col}. Got: {cols}")

        has_file = file_col in cols if file_col else False

        for row in r:
            # preserve all original fields as strings
            base = {k: ("" if row.get(k) is None else str(row.get(k))) for k in cols}

            raw = filter_ascii_printable(str(row.get(raw_col, "") or ""), keep_space=True)
            clean = filter_ascii_printable(str(row.get(clean_col, "") or ""), keep_space=True)

            raw = raw[:max_len] if max_len > 0 else raw
            clean = clean  # do not truncate clean; alignment uses clean as reference

            if not raw:
                continue

            base["_raw_norm"] = raw
            base["_clean_norm"] = clean
            if has_file:
                base["_file"] = str(row.get(file_col, "") or "")
            rows.append(base)

    return rows


def split_rows(
    rows: List[Dict[str, str]],
    val_ratio: float,
    seed: int,
    split_by_file: bool,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)

    if split_by_file and rows and "_file" in rows[0]:
        by_file: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            by_file.setdefault(r.get("_file", ""), []).append(r)

        keys = list(by_file.keys())
        rng.shuffle(keys)

        total = len(rows)
        target_val = int(round(total * val_ratio))
        val_rows: List[Dict[str, str]] = []
        train_rows: List[Dict[str, str]] = []

        cur_val = 0
        for k in keys:
            grp = by_file[k]
            if cur_val < target_val:
                val_rows.extend(grp)
                cur_val += len(grp)
            else:
                train_rows.extend(grp)

        if not train_rows:
            train_rows, val_rows = rows[:-1], rows[-1:]
        return train_rows, val_rows

    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n_val = int(round(len(rows) * val_ratio))
    val_set = set(idxs[:n_val])
    train_rows = [rows[i] for i in idxs[n_val:]]
    val_rows = [rows[i] for i in idxs[:n_val]]
    if not train_rows:
        train_rows, val_rows = rows[:-1], rows[-1:]
    return train_rows, val_rows


def build_vocab_from_rows(rows: List[Dict[str, str]], max_vocab: int = 0) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for r in rows:
        raw = r["_raw_norm"]
        for ch in raw:
            freq[ch] = freq.get(ch, 0) + 1

    chars = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    if max_vocab and max_vocab > 0:
        chars = chars[:max_vocab]

    vocab = {PAD: 0, UNK: 1}
    for ch, _ in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def save_vocab_json(vocab: Dict[str, int], out_path: str) -> None:
    inv = {str(k): int(v) for k, v in vocab.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved vocab: {out_path}")


# -----------------------------
# Dataset / Collate
# -----------------------------
class KeepDropRows(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, str]],
        vocab: Dict[str, int],
        casefold: bool,
        min_cov_clean: float,
    ):
        self.rows_in = rows
        self.vocab = vocab
        self.casefold = casefold
        self.min_cov_clean = float(min_cov_clean)

        self.rows: List[Tuple[str, str]] = []
        dropped = 0

        for r in rows:
            raw = r["_raw_norm"]
            clean = r["_clean_norm"]

            ok, mask = greedy_subseq_keep_mask(raw, clean, casefold=self.casefold)
            if not ok:
                mask = lcs_keep_mask(raw, clean, casefold=self.casefold)
                lcs_len = float(sum(mask))
                cov_clean = lcs_len / max(1.0, float(len(maybe_casefold(clean, self.casefold))))
                if cov_clean < self.min_cov_clean:
                    dropped += 1
                    continue

            self.rows.append((raw, clean))

        logger.info(f"Dataset built: kept={len(self.rows)} dropped={dropped} (min_cov_clean={self.min_cov_clean})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        raw, clean = self.rows[idx]

        ok, mask = greedy_subseq_keep_mask(raw, clean, casefold=self.casefold)
        if not ok:
            mask = lcs_keep_mask(raw, clean, casefold=self.casefold)

        ids = [self.vocab.get(ch, self.vocab[UNK]) for ch in raw]
        y = mask
        return raw, clean, ids, y


@dataclass
class Batch:
    raw: List[str]
    clean: List[str]
    input_ids: torch.Tensor  # (B,T)
    labels: torch.Tensor     # (B,T)
    attn_mask: torch.Tensor  # (B,T) float 1=valid,0=pad


def collate_fn(batch, pad_id: int) -> Batch:
    raw_list, clean_list, ids_list, y_list = zip(*batch)
    max_t = max(len(x) for x in ids_list)

    input_ids = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    labels = torch.zeros((len(batch), max_t), dtype=torch.float32)
    attn_mask = torch.zeros((len(batch), max_t), dtype=torch.float32)

    for i, (ids, y) in enumerate(zip(ids_list, y_list)):
        t = len(ids)
        input_ids[i, :t] = torch.tensor(ids, dtype=torch.long)
        labels[i, :t] = torch.tensor(y, dtype=torch.float32)
        attn_mask[i, :t] = 1.0

    return Batch(
        raw=list(raw_list),
        clean=list(clean_list),
        input_ids=input_ids,
        labels=labels,
        attn_mask=attn_mask,
    )


# -----------------------------
# Model
# -----------------------------
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


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def token_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, thr: float = 0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).float()

    valid = (mask > 0.5)
    pred_v = pred[valid]
    lab_v = labels[valid]
    if pred_v.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0

    tp = ((pred_v == 1) & (lab_v == 1)).sum().item()
    fp = ((pred_v == 1) & (lab_v == 0)).sum().item()
    fn = ((pred_v == 0) & (lab_v == 1)).sum().item()
    tn = ((pred_v == 0) & (lab_v == 0)).sum().item()

    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    return float(acc), float(prec), float(rec), float(f1)


@torch.no_grad()
def exact_match_rate(raw_list: List[str], clean_list: List[str], logits: torch.Tensor, mask: torch.Tensor,
                     casefold: bool, thr: float = 0.5) -> float:
    probs = torch.sigmoid(logits).cpu()
    pred = (probs >= thr).cpu()
    mask = mask.cpu()

    hit = 0
    for i, (raw, clean) in enumerate(zip(raw_list, clean_list)):
        out_chars = []
        for j in range(len(raw)):
            if mask[i, j].item() < 0.5:
                break
            if pred[i, j].item():
                out_chars.append(raw[j])
        out = "".join(out_chars)
        if maybe_casefold(out, casefold) == maybe_casefold(clean, casefold):
            hit += 1
    return hit / max(1, len(raw_list))


# -----------------------------
# Lightning DataModule
# -----------------------------
class KeepDropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
        max_len: int,
        casefold: bool,
        max_vocab: int,
        raw_col: str,
        clean_col: str,
        file_col: str,
        split_by_file: bool,
        val_ratio: float,
        seed: int,
        min_cov_clean: float,
        num_workers: int = 0,
        val_path: str = "",
    ):
        super().__init__()
        self.data_path = data_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.casefold = casefold
        self.max_vocab = max_vocab
        self.raw_col = raw_col
        self.clean_col = clean_col
        self.file_col = file_col
        self.split_by_file = split_by_file
        self.val_ratio = val_ratio
        self.seed = seed
        self.min_cov_clean = min_cov_clean
        self.num_workers = num_workers

        self.vocab: Dict[str, int] = {}
        self.all_rows: List[Dict[str, str]] = []
        self.train_rows: List[Dict[str, str]] = []
        self.val_rows: List[Dict[str, str]] = []
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str | None = None):
        self.all_rows = load_rows_from_csv(
            self.data_path,
            raw_col=self.raw_col,
            clean_col=self.clean_col,
            file_col=self.file_col,
            max_len=self.max_len,
        )
        if not self.all_rows:
            raise SystemExit("No valid rows loaded from CSV (raw empty after filtering?)")

        if self.val_path:
            self.train_rows = self.all_rows
            self.val_rows = load_rows_from_csv(self.val_path, self.raw_col, self.clean_col, self.file_col, self.max_len)
            if not self.val_rows:
                raise ValueError("Validation data is empty")
            train_files = {r.get("_file") for r in self.train_rows}
            val_files = {r.get("_file") for r in self.val_rows}
            if None in train_files or None in val_files or "" in train_files or "" in val_files or train_files & val_files:
                raise ValueError("Explicit train/validation groups must be present and disjoint")
        else:
            self.train_rows, self.val_rows = split_rows(
                self.all_rows, self.val_ratio, self.seed, self.split_by_file,
            )
        logger.info(f"Split: train_rows={len(self.train_rows)} val_rows={len(self.val_rows)} (val_ratio={self.val_ratio}, split_by_file={self.split_by_file})")

        if not self.vocab:
            self.vocab = build_vocab_from_rows(self.train_rows, max_vocab=self.max_vocab)
            logger.info(f"vocab_size={len(self.vocab)} (PAD=0, UNK=1)")

        if stage in (None, "fit"):
            self.train_ds = KeepDropRows(self.train_rows, vocab=self.vocab, casefold=self.casefold, min_cov_clean=self.min_cov_clean)
            self.val_ds = KeepDropRows(self.val_rows, vocab=self.vocab, casefold=self.casefold, min_cov_clean=self.min_cov_clean)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_fn(b, pad_id=self.vocab[PAD]),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_fn(b, pad_id=self.vocab[PAD]),
        )


# -----------------------------
# Lightning Module
# -----------------------------
class KeepDropLightning(pl.LightningModule):
    def __init__(
        self,
        vocab: Dict[str, int],
        emb_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        casefold: bool,
        thr: float = 0.5,
        log_val_samples: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["vocab"])
        self.vocab = vocab
        self.model = CharKeepDropLSTM(
            vocab_size=len(vocab),
            emb_dim=emb_dim,
            hidden=hidden,
            num_layers=layers,
            dropout=dropout,
        )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def training_step(self, batch: Batch, batch_idx: int):
        logits = self(batch.input_ids)
        loss_raw = self.bce(logits, batch.labels)
        loss = (loss_raw * batch.attn_mask).sum() / batch.attn_mask.sum().clamp_min(1.0)

        acc, prec, rec, f1 = token_metrics_from_logits(logits, batch.labels, batch.attn_mask, thr=self.hparams.thr)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch.raw))
        self.log("train/f1", f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("train/prec", prec, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("train/rec", rec, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("train/acc", acc, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        logits = self(batch.input_ids)
        loss_raw = self.bce(logits, batch.labels)
        loss = (loss_raw * batch.attn_mask).sum() / batch.attn_mask.sum().clamp_min(1.0)

        acc, prec, rec, f1 = token_metrics_from_logits(logits, batch.labels, batch.attn_mask, thr=self.hparams.thr)
        em = exact_match_rate(batch.raw, batch.clean, logits, batch.attn_mask, casefold=self.hparams.casefold, thr=self.hparams.thr)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("val/f1", f1, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("val/exact", em, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("val/prec", prec, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("val/rec", rec, on_step=False, on_epoch=True, batch_size=len(batch.raw))
        self.log("val/acc", acc, on_step=False, on_epoch=True, batch_size=len(batch.raw))

        if batch_idx == 0 and self.hparams.log_val_samples > 0:
            probs = torch.sigmoid(logits).detach().cpu()
            pred = (probs >= self.hparams.thr).cpu()
            for i in range(min(self.hparams.log_val_samples, len(batch.raw))):
                raw = batch.raw[i]
                out = "".join(ch for ch, k in zip(raw, pred[i].tolist()) if k)
                logger.info(f"[VAL SAMPLE] raw:   {raw}")
                logger.info(f"[VAL SAMPLE] clean: {batch.clean[i]}")
                logger.info(f"[VAL SAMPLE] pred:  {out}")

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip = float(self.hparams.grad_clip)
        if clip and clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=clip, gradient_clip_algorithm="norm")


# -----------------------------
# Post-train export
# -----------------------------
@torch.no_grad()
def predict_all_and_export(
    best_ckpt_path: str,
    vocab: Dict[str, int],
    data_rows: List[Dict[str, str]],
    out_csv_path: str,
    device: torch.device,
    thr: float,
    max_len: int,
    raw_col: str,
    clean_col: str,
):
    """
    Run model on ALL rows in the original CSV and export CSV with extra column `pred`.
    Uses normalized raw used by training (`_raw_norm`).
    """
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise SystemExit(f"Checkpoint not found: {best_ckpt_path}")

    model = KeepDropLightning.load_from_checkpoint(best_ckpt_path, vocab=vocab)
    model.eval()
    model.to(device)

    # prepare writer columns: original columns + pred + pred_from_norm + used_raw_len
    orig_cols = [c for c in data_rows[0].keys() if not c.startswith("_")]
    extra_cols = ["pred", "pred_norm", "used_raw_len"]
    fieldnames = orig_cols + extra_cols

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    pad_id = vocab[PAD]
    unk_id = vocab[UNK]

    def encode(raw_norm: str) -> torch.Tensor:
        ids = [vocab.get(ch, unk_id) for ch in raw_norm]
        return torch.tensor(ids, dtype=torch.long)

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # small batching for speed
        B = 256
        for start in range(0, len(data_rows), B):
            chunk = data_rows[start:start + B]
            raws_norm = [r["_raw_norm"][:max_len] for r in chunk]

            # pad batch
            max_t = max(len(s) for s in raws_norm) if raws_norm else 0
            input_ids = torch.full((len(chunk), max_t), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), max_t), dtype=torch.float32)
            for i, s in enumerate(raws_norm):
                ids = encode(s)
                t = ids.numel()
                input_ids[i, :t] = ids
                attn[i, :t] = 1.0

            input_ids = input_ids.to(device)
            logits = model(input_ids)  # (B,T)
            probs = torch.sigmoid(logits).cpu()
            keep = (probs >= thr)

            for i, r in enumerate(chunk):
                raw_orig = str(r.get(raw_col, ""))  # as stored in CSV
                raw_norm = r["_raw_norm"][:max_len]
                kept_chars = [ch for ch, k in zip(raw_norm, keep[i].tolist()) if k]
                pred_norm = "".join(kept_chars)

                out_row = {k: r.get(k, "") for k in orig_cols}
                out_row["pred"] = pred_norm  # 这里 pred 就用规范后的 raw 推理结果
                out_row["pred_norm"] = pred_norm
                out_row["used_raw_len"] = str(len(raw_norm))
                w.writerow(out_row)

    logger.info(f"Exported predictions for all rows: {out_csv_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Single CSV path (will auto split into train/val)")
    ap.add_argument("--val_data", default="", help="Frozen validation CSV; skips random splitting")
    ap.add_argument("--raw_col", default="raw", help="CSV column name for raw")
    ap.add_argument("--clean_col", default="clean", help="CSV column name for clean")
    ap.add_argument("--file_col", default="file", help="CSV column name for grouping split (default: file)")
    ap.add_argument("--split_by_file", action="store_true", help="Group split by file column to avoid leakage")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    ap.add_argument("--min_cov_clean", type=float, default=1.0,
                    help="Drop samples if LCS(clean,raw)/len(clean) < this when greedy subseq fails.")

    ap.add_argument("--outdir", default="runs/keepdrop_lightning", help="Output dir")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_vocab", type=int, default=0)
    ap.add_argument("--casefold", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--thr", type=float, default=0.5, help="keep threshold for prediction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", default="32", help="Trainer precision, e.g. 32 / 16-mixed / bf16-mixed")
    ap.add_argument("--devices", default="auto", help="Trainer devices, e.g. auto / 1 / 0,1")
    ap.add_argument("--accelerator", default="auto", help="auto/cpu/gpu")
    ap.add_argument("--resume", default="", help="resume from ckpt path (Lightning ckpt)")

    ap.add_argument("--export_all", action="store_true", help="After training, run best checkpoint over ALL CSV rows and export")
    ap.add_argument("--train_only", action="store_true", help="Fixed-epoch fit on all development data")
    args = ap.parse_args()
    if args.train_only and args.val_data:
        ap.error("--train_only cannot use validation data")

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    dm = KeepDropDataModule(
        data_path=args.data,
        val_path=args.val_data,
        batch_size=args.batch_size,
        max_len=args.max_len,
        casefold=args.casefold,
        max_vocab=args.max_vocab,
        raw_col=args.raw_col,
        clean_col=args.clean_col,
        file_col=args.file_col,
        split_by_file=args.split_by_file,
        val_ratio=0.0 if args.train_only else args.val_ratio,
        seed=args.seed,
        min_cov_clean=args.min_cov_clean,
        num_workers=args.num_workers,
    )
    dm.setup("fit")

    model = KeepDropLightning(
        vocab=dm.vocab,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        casefold=args.casefold,
        thr=args.thr,
        log_val_samples=3,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.outdir, "checkpoints"),
        filename="best-{epoch:02d}",
        monitor=None if args.train_only else "val/exact",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    tb_logger = TensorBoardLogger(save_dir=args.outdir, name="tb")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=tb_logger,
        callbacks=[lr_cb] if args.train_only else [ckpt_cb, lr_cb],
        enable_checkpointing=not args.train_only,
        limit_val_batches=0 if args.train_only else 1.0,
        num_sanity_val_steps=0 if args.train_only else 2,
        log_every_n_steps=20,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else 0.0,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume if args.resume else None)

    if args.train_only:
        trainer.save_checkpoint(os.path.join(args.outdir, "model.ckpt"))
    best_path = os.path.join(args.outdir, "model.ckpt") if args.train_only else ckpt_cb.best_model_path
    last_path = ckpt_cb.last_model_path
    logger.info(f"Best ckpt: {best_path}")
    logger.info(f"Last ckpt: {last_path}")

    # Save vocab for reproducible inference
    save_vocab_json(dm.vocab, os.path.join(args.outdir, "vocab.json"))

    if args.export_all:
        ckpt_to_use = best_path if best_path else last_path
        if not ckpt_to_use:
            raise SystemExit("No checkpoint produced (best/last empty).")

        device = torch.device("cuda" if torch.cuda.is_available() and args.accelerator != "cpu" else "cpu")
        out_csv = os.path.join(args.outdir, "pred_all.csv")

        predict_all_and_export(
            best_ckpt_path=ckpt_to_use,
            vocab=dm.vocab,
            data_rows=dm.all_rows,   # ALL rows (after filtering/truncation rules)
            out_csv_path=out_csv,
            device=device,
            thr=float(args.thr),
            max_len=int(args.max_len),
            raw_col=args.raw_col,
            clean_col=args.clean_col,
        )


if __name__ == "__main__":
    main()