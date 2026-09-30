#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

# Optional metrics
try:
    import evaluate

    SEQEVAL = evaluate.load("seqeval")
except Exception:
    SEQEVAL = None


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def poly_to_box(poly: List[List[float]], w: int, h: int) -> List[int]:
    """
    Convert 4-point polygon to axis-aligned rectangle bbox: [x0,y0,x1,y1]
    then clamp to image bounds.
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0 = int(max(0, min(xs)))
    y0 = int(max(0, min(ys)))
    x1 = int(min(w - 1, max(xs)))
    y1 = int(min(h - 1, max(ys)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def normalize_box(box: List[int], w: int, h: int) -> List[int]:
    """
    Normalize bbox from pixel coords to LayoutLM expected scale [0,1000].
    """
    x0, y0, x1, y1 = box
    return [
        int(1000 * x0 / w),
        int(1000 * y0 / h),
        int(1000 * x1 / w),
        int(1000 * y1 / h),
    ]


def find_samples(data_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Return list of (image_path, ocr_json_path, label_json_path)
    Only include if all exist.
    """
    samples = []
    for img_path in sorted(data_dir.glob("*.jpeg")):
        stem = img_path.stem
        ocr_path = data_dir / f"{stem}-ocr.json"
        label_path = data_dir / f"{stem}-label.json"
        if ocr_path.exists() and label_path.exists():
            samples.append((img_path, ocr_path, label_path))
    return samples


def scan_all_labels(samples: List[Tuple[Path, Path, Path]]) -> List[str]:
    """
    Scan label_json to collect all label strings.
    Add "O" as default label.
    Only count label files that are valid JSON dict.
    """
    label_set = set()
    for _, _, label_path in samples:
        try:
            obj = read_json(label_path)
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str) and v.strip():
                        label_set.add(v.strip())
        except Exception:
            continue
    label_set.add("O")
    labels = sorted(label_set)
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    return labels


def load_doc(
    ocr_path: Path,
    label_path: Path,
    image: Image.Image,
) -> Optional[Tuple[List[str], List[List[int]], List[str]]]:
    """
    Build (words, boxes, word_labels) for one image.
    Return None if no valid labels in label.json.
    """
    ocr = read_json(ocr_path)
    lbl = read_json(label_path)

    if not isinstance(ocr, dict) or not isinstance(lbl, dict):
        return None

    rec_texts = ocr.get("rec_texts", None)
    rec_polys = ocr.get("rec_polys", None)

    if not isinstance(rec_texts, list) or not isinstance(rec_polys, list):
        return None
    if len(rec_texts) != len(rec_polys):
        n = min(len(rec_texts), len(rec_polys))
        rec_texts = rec_texts[:n]
        rec_polys = rec_polys[:n]

    w, h = image.size
    words: List[str] = []
    boxes: List[List[int]] = []
    word_labels: List[str] = []

    valid_labeled_indices = []
    for k, v in lbl.items():
        try:
            idx = int(k)
        except Exception:
            continue
        if 0 <= idx < len(rec_texts) and isinstance(v, str) and v.strip():
            valid_labeled_indices.append(idx)

    if len(valid_labeled_indices) == 0:
        return None

    for i, (t, p) in enumerate(zip(rec_texts, rec_polys)):
        if t is None:
            t = ""
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if t == "":
            continue

        if not (isinstance(p, list) and len(p) == 4):
            continue

        rect = poly_to_box(p, w, h)
        nbox = normalize_box(rect, w, h)

        words.append(t)
        boxes.append(nbox)

        lab = lbl.get(str(i), "O")
        if not isinstance(lab, str) or not lab.strip():
            lab = "O"
        word_labels.append(lab.strip())

    if len(words) == 0:
        return None

    return words, boxes, word_labels


class OcrKIEDataset(torch.utils.data.Dataset):
    """
    IMPORTANT:
    - With return_overflowing_tokens=True, one document can produce multiple chunks.
    - We "flatten" all chunks into self.examples so Trainer sees a normal dataset.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, Path, Path]],
        processor: LayoutLMv3Processor,
        label2id: Dict[str, int],
        max_length: int = 512,
        stride: int = 128,
    ):
        self.processor = processor
        self.label2id = label2id
        self.max_length = max_length
        self.stride = stride

        self.examples: List[Dict[str, torch.Tensor]] = []

        for img_path, ocr_path, label_path in samples:
            try:
                image = Image.open(img_path).convert("RGB")
                doc = load_doc(ocr_path, label_path, image)
                if doc is None:
                    continue
                words, boxes, word_labels = doc
                word_label_ids = [self.label2id.get(l, self.label2id["O"]) for l in word_labels]

                enc = self.processor(
                    image,
                    text=words,
                    boxes=boxes,
                    word_labels=word_label_ids,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    stride=self.stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=False,  # training: not needed
                    return_tensors="pt",
                )

                # enc is batch = num_spans
                # Keep only tensor fields (avoid any non-tensor mapping keys just in case)
                tensor_keys = [k for k in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels") if k in enc and isinstance(enc[k], torch.Tensor)]
                if not tensor_keys:
                    continue

                num_spans = enc[tensor_keys[0]].shape[0]
                for i in range(num_spans):
                    ex = {k: enc[k][i] for k in tensor_keys}
                    self.examples.append(ex)
            except Exception as error:
                raise ValueError(f"Invalid training sample: {img_path.name}") from error

        if len(self.examples) == 0:
            raise RuntimeError("No valid training samples found after filtering / chunking.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def compute_metrics_fn(id2label: Dict[int, str]):
    """
    seqeval expects list of label strings per sentence (token-level),
    with special tokens removed (-100).
    """
    if SEQEVAL is None:
        return None

    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(preds, labels):
            sent_preds = []
            sent_labels = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                sent_preds.append(id2label[int(p)])
                sent_labels.append(id2label[int(l)])
            true_predictions.append(sent_preds)
            true_labels.append(sent_labels)

        results = SEQEVAL.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    return _compute


def train_val_split(items: List[Tuple[Path, Path, Path]], val_ratio: float, seed: int):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n_val = int(round(len(items) * val_ratio))
    val = items[:n_val]
    train = items[n_val:]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="label_img",
                    help="Folder containing *.jpeg and *-ocr.json/*-label.json")
    ap.add_argument("--val_dir", type=str, default="", help="Frozen validation directory; bypass random split")
    ap.add_argument("--output_dir", type=str, default="layoutlmv3_out")
    ap.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128, help="sliding window stride in tokens (used with overflow)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_bs", type=int, default=3)
    ap.add_argument("--eval_bs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_only", action="store_true", help="Fixed-epoch fit on all development data; no checkpoint selection")
    args = ap.parse_args()
    if args.train_only and args.val_dir:
        ap.error("--train_only cannot use validation data")
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    samples = find_samples(data_dir)
    if len(samples) == 0:
        raise RuntimeError(
            f"No samples found in {data_dir}. Expect *.jpeg with matching *-ocr.json and *-label.json"
        )

    labels = ["O"] + [f"{prefix}-{kind}" for prefix in ("B", "I") for kind in ("QTY", "RECIPIENT", "SKU", "TRACK")]
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    print(f"[INFO] Found {len(labels)} labels:")
    for lab in labels:
        print("  ", lab)

    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)

    if args.train_only:
        train_samples, val_samples = samples, []
    elif args.val_dir:
        train_samples, val_samples = samples, find_samples(Path(args.val_dir))
        if not val_samples or {p[0].stem for p in train_samples} & {p[0].stem for p in val_samples}:
            raise ValueError("Explicit validation data must be nonempty and disjoint")
    else:
        train_samples, val_samples = train_val_split(samples, args.val_ratio, args.seed)

    train_ds = OcrKIEDataset(
        train_samples,
        processor,
        label2id,
        max_length=args.max_length,
        stride=args.stride,
    )
    val_ds = None if args.train_only else OcrKIEDataset(
        val_samples,
        processor,
        label2id,
        max_length=args.max_length,
        stride=args.stride,
    )

    print(f"[INFO] Flattened train chunks: {len(train_ds)}")
    print(f"[INFO] Flattened val chunks:   {len(val_ds) if val_ds is not None else 0}")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="no" if args.train_only else "epoch",
        save_strategy="no" if args.train_only else "epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=not args.train_only,
        metric_for_best_model="f1" if SEQEVAL is not None else "eval_loss",
        greater_is_better=True if SEQEVAL is not None else False,
        seed=args.seed,
        fp16=True,
        report_to="none",
    )

    metrics_fn = compute_metrics_fn(id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,  # used mainly for saving
        data_collator=default_data_collator,
        compute_metrics=metrics_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"[DONE] Saved model & processor to: {args.output_dir}")


if __name__ == "__main__":
    main()
