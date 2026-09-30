"""LayoutLMv3 inference, preserved from nga-torch 5d0556a."""
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
logger = logging.getLogger(__name__)

def poly_to_rect(poly: List[List[float]], w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]
    x0 = int(max(0, min(xs)))
    y0 = int(max(0, min(ys)))
    x1 = int(min(w - 1, max(xs)))
    y1 = int(min(h - 1, max(ys)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1

def normalize_box(rect: Tuple[int, int, int, int], w: int, h: int) -> List[int]:
    x0, y0, x1, y1 = rect

    def clip(v: int) -> int:
        return max(0, min(1000, v))

    return [
        clip(int(1000 * x0 / w)),
        clip(int(1000 * y0 / h)),
        clip(int(1000 * x1 / w)),
        clip(int(1000 * y1 / h)),
    ]

def build_items_from_ocr_res(
        ocr_res: dict, img_bgr: np.ndarray
) -> Tuple[List[int], List[str], List[List[int]], List[List[List[float]]], List[Tuple[int, int, int, int]]]:
    rec_texts = ocr_res.get("rec_texts", [])
    rec_polys = ocr_res.get("rec_polys", [])
    if not isinstance(rec_texts, list) or not isinstance(rec_polys, list):
        raise ValueError("ocr_res missing/invalid rec_texts or rec_polys")

    n = min(len(rec_texts), len(rec_polys))
    rec_texts = rec_texts[:n]
    rec_polys = rec_polys[:n]

    h, w = img_bgr.shape[:2]

    kept_indices: List[int] = []
    words: List[str] = []
    boxes_norm: List[List[int]] = []
    polys: List[List[List[float]]] = []
    rects_px: List[Tuple[int, int, int, int]] = []

    for idx, (t, poly) in enumerate(zip(rec_texts, rec_polys)):
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        t = t.strip()
        if t == "":
            continue
        if not (isinstance(poly, list) and len(poly) == 4):
            continue

        rect = poly_to_rect(poly, w, h)
        kept_indices.append(idx)
        words.append(t)
        polys.append(poly)
        rects_px.append(rect)
        boxes_norm.append(normalize_box(rect, w, h))

    if not words:
        raise ValueError("No valid OCR words after filtering empties.")

    return kept_indices, words, boxes_norm, polys, rects_px

def bgr_from_label_palette(labels: List[str]) -> Dict[str, Tuple[int, int, int]]:
    rgb_palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    bgr_palette = [(b, g, r) for (r, g, b) in rgb_palette]
    uniq = sorted({l for l in labels if l != "O"})
    return {lab: bgr_palette[i % len(bgr_palette)] for i, lab in enumerate(uniq)}

def draw_overlay(
        img_bgr: np.ndarray,
        rects_px: List[Tuple[int, int, int, int]],
        pred_labels: List[str],
        label_colors: Dict[str, Tuple[int, int, int]],
        alpha_o: float = 0.22,
        alpha_t: float = 0.45,
) -> np.ndarray:
    out = img_bgr.copy()

    for (x0, y0, x1, y1), lab in zip(rects_px, pred_labels):
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(out.shape[1] - 1, x1), min(out.shape[0] - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        if lab == "O":
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
            out = cv2.addWeighted(overlay, alpha_o, out, 1 - alpha_o, 0)
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
        else:
            color = label_colors.get(lab, (0, 255, 255))
            overlay = out.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
            out = cv2.addWeighted(overlay, alpha_t, out, 1 - alpha_t, 0)
            cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(lab, font, scale, thickness)
            ty = max(y0 - 4, th + 2)
            tx = x0

            bg_x1 = min(out.shape[1] - 1, tx + tw + 6)
            bg_y0 = max(0, ty - th - 6)
            bg_y1 = min(out.shape[0] - 1, ty + 2)

            overlay2 = out.copy()
            cv2.rectangle(overlay2, (tx, bg_y0), (bg_x1, bg_y1), color, thickness=-1)
            out = cv2.addWeighted(overlay2, 0.70, out, 0.30, 0)
            cv2.putText(out, lab, (tx + 3, ty - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return out

class LayoutLMRunner:
    def __init__(
            self,
            checkpoint: Path,
            max_length: int,
            stride: int,
            topk: int,
            device: torch.device,
            infer_span_batch: int = 8,
            use_amp: bool = True,
    ):
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.stride = stride
        self.topk = topk
        self.device = device
        self.infer_span_batch = max(1, int(infer_span_batch))
        self.use_amp = bool(use_amp)

        self.processor = LayoutLMv3Processor.from_pretrained(str(checkpoint), apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(checkpoint)).to(device)
        self.model.eval()

        # 保险：config 里的 key 有时是 str
        self.id2label: Dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id: Dict[str, int] = {k: int(v) for k, v in self.model.config.label2id.items()}

        logger.info(
            "Loaded model. checkpoint=%s device=%s num_labels=%s max_length=%s stride=%s",
            str(checkpoint),
            str(device),
            str(getattr(self.model.config, "num_labels", "?")),
            str(self.max_length),
            str(self.stride),
        )

    def infer_with_meta(
            self,
            img_bgr: np.ndarray,
            kept_indices: List[int],
            words: List[str],
            boxes_norm: List[List[int]],
            polys: List[List[List[float]]],
            rects_px: List[Tuple[int, int, int, int]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        t0 = time.time()
        encoding = self.processor(
            pil_img,
            text=words,
            boxes=boxes_norm,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,  # 推理不需要
            return_tensors="pt",
        )

        # num_spans
        if "input_ids" not in encoding or not isinstance(encoding["input_ids"], torch.Tensor):
            raise RuntimeError("processor output missing input_ids tensor")
        num_spans = int(encoding["input_ids"].shape[0])

        num_labels = int(getattr(self.model.config, "num_labels", 0)) or len(self.id2label)
        if num_labels <= 0:
            raise RuntimeError("invalid num_labels from model config")

        # 汇总：每个 word 一个 label 概率向量，来自所有 spans 的 element-wise max
        # 用 CPU tensor 存，避免占 GPU（word 数量一般不大）
        word_score = torch.zeros((len(words), num_labels), dtype=torch.float32)
        word_best_conf = torch.zeros((len(words),), dtype=torch.float32)

        # helper: 某个 span 的 word_ids
        def _get_word_ids(span_idx: int):
            try:
                return encoding.word_ids(batch_index=span_idx)
            except Exception:
                # 兼容旧路径
                try:
                    return encoding.encodings[span_idx].word_ids  # type: ignore[attr-defined]
                except Exception:
                    return None

        # 分批 forward，防 spans 太多 OOM
        with torch.no_grad():
            for s0 in range(0, num_spans, self.infer_span_batch):
                s1 = min(num_spans, s0 + self.infer_span_batch)
                MODEL_KEYS = ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids")

                enc_dev = {}
                for k in MODEL_KEYS:
                    v = encoding.get(k, None)
                    if isinstance(v, torch.Tensor):
                        enc_dev[k] = v[s0:s1].to(self.device, non_blocking=True)

                if self.device.type == "cuda" and self.use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = self.model(**enc_dev)
                else:
                    out = self.model(**enc_dev)

                logits = out.logits  # [b, seq, num_labels]
                probs = torch.softmax(logits.float(), dim=-1).cpu()  # 回 CPU 做汇总

                bsz = probs.shape[0]
                for bi in range(bsz):
                    span_idx = s0 + bi
                    word_ids = _get_word_ids(span_idx)
                    if word_ids is None:
                        raise RuntimeError("Could not obtain word_ids; cannot map token preds back to words.")

                    seen = set()
                    for tok_idx, wid in enumerate(word_ids):
                        if wid is None or wid in seen:
                            continue
                        seen.add(wid)
                        if not (0 <= int(wid) < len(words)):
                            continue

                        pv = probs[bi, tok_idx]  # [num_labels]
                        # element-wise max accumulate
                        prev = word_score[wid]
                        word_score[wid] = torch.maximum(prev, pv)

                        # 记录一下这个 token 的 max prob，用于 debug/置信度备选
                        c = float(pv.max().item())
                        if c > float(word_best_conf[wid].item()):
                            word_best_conf[wid] = float(c)

        # 输出最终 word 级 pred + conf + topk
        word_pred: List[str] = ["O"] * len(words)
        word_conf: List[float] = [0.0] * len(words)
        word_topk: List[List[Tuple[str, float]]] = [[] for _ in range(len(words))]

        for i in range(len(words)):
            scores = word_score[i]
            if float(scores.sum().item()) <= 0.0:
                # fallback
                word_pred[i] = "O"
                word_conf[i] = 0.0
                word_topk[i] = [("O", 0.0)]
                continue

            lid = int(torch.argmax(scores).item())
            lab = self.id2label.get(lid, "O")
            conf = float(scores[lid].item())

            tk = min(self.topk, scores.numel())
            top_vals, top_ids = torch.topk(scores, k=tk)
            tk_list = []
            for p, idx in zip(top_vals.tolist(), top_ids.tolist()):
                tk_list.append((self.id2label.get(int(idx), str(int(idx))), float(p)))

            word_pred[i] = lab
            word_conf[i] = conf
            word_topk[i] = tk_list

        infer_elapsed = time.time() - t0

        non_o_positions = [i for i, l in enumerate(word_pred) if l != "O"]
        non_o_indices = [int(kept_indices[i]) for i in non_o_positions]

        # Non-O first (stable)
        order = sorted(range(len(words)), key=lambda j: (word_pred[j] == "O", j))

        items = []
        for j in order:
            rect = rects_px[j]
            items.append(
                {
                    "ocr_index": int(kept_indices[j]),
                    "text": words[j],
                    "pred": {
                        "label": word_pred[j],
                        "label_id": int(self.label2id.get(word_pred[j], -1)),
                        "confidence": float(word_conf[j]),
                        "topk": [{"label": lab, "prob": float(p)} for lab, p in word_topk[j]],
                    },
                    "bbox": {
                        "poly": polys[j],
                        "rect_px": [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])],
                        "box_1000": [int(x) for x in boxes_norm[j]],
                    },
                }
            )

        checkpoint_block = {
            "name": self.checkpoint.name,
            "path": str(self.checkpoint),
            "infer_elapsed_sec": float(infer_elapsed),
            "non_o_count": int(len(non_o_positions)),
            "kept_count": int(len(words)),
            "num_spans": int(num_spans),
            "stride": int(self.stride),
            "items": items,
            "non_o_indices": non_o_indices,
        }

        return checkpoint_block, word_pred
