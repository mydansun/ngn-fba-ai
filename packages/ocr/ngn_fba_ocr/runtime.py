"""Pretrained PaddleOCR; no OCR training required."""
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult

def create_ocr(precision: str = "fp32", device: str = "gpu:0") -> PaddleOCR:
    """统一创建 PaddleOCR 实例的参数"""
    return PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        precision=precision,
        device=device,
    )

def extract_ocr_res_and_output(result: List[OCRResult]) -> Tuple[Dict[str, Any], np.ndarray]:
    """从 PaddleOCR 的结果列表中提取核心 JSON 和输出图像"""
    if not result:
        raise RuntimeError("OCR returned empty result list.")

    res0 = result[0]
    j = getattr(res0, "json", None)
    if not isinstance(j, dict) or "res" not in j or not isinstance(j["res"], dict):
        raise RuntimeError("OCRResult.json['res'] missing or invalid.")

    output_img_bgr = res0["doc_preprocessor_res"]["output_img"]
    if isinstance(output_img_bgr, list):
        output_img_bgr = np.array(output_img_bgr)

    if not isinstance(output_img_bgr, np.ndarray):
        raise RuntimeError("doc_preprocessor_res.output_img is not ndarray.")

    if output_img_bgr.dtype != np.uint8:
        output_img_bgr = output_img_bgr.astype(np.uint8, copy=False)
    output_img_bgr = np.ascontiguousarray(output_img_bgr)

    return j["res"], output_img_bgr

def bgr_ndarray_to_jpeg_bytes(bgr_img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr_img)
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buf.tobytes()
