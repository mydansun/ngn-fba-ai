import json
import math
import os
from pathlib import Path
import cv2
import torch
from ngn_fba_ai_proto import ai_pb2 as pb, ai_pb2_grpc as rpc
from ngn_fba_ai_proto.server import serve, decode_image
from .runtime import LayoutLMRunner, build_items_from_ocr_res, draw_overlay, bgr_from_label_palette


class Service(rpc.LayoutServiceServicer):
    def __init__(self):
        torch.set_num_threads(4)
        self.runner = LayoutLMRunner(Path(os.environ['MODEL_DIR']), 512, 128, 3, torch.device('cuda'), 4, True)

    def Health(self, request, context):
        return pb.HealthResponse(model=os.getenv('MODEL_VERSION', 'final-fit-v1'))

    def Predict(self, request, context):
        image = decode_image(request.image)
        if len(request.ocr_json) > 1024 * 1024:
            raise ValueError('OCR payload too large')
        ocr = json.loads(request.ocr_json)
        if not isinstance(ocr, dict): raise ValueError('Invalid OCR object')
        texts, polygons = ocr.get('rec_texts'), ocr.get('rec_polys')
        if not isinstance(texts, list) or not isinstance(polygons, list) or len(texts) != len(polygons) or len(texts) > 1000:
            raise ValueError('Invalid OCR arrays')
        if not all(isinstance(t, str) and len(t) <= 4096 for t in texts) or sum(map(len, texts)) > 32768:
            raise ValueError('Invalid OCR text')
        h, w = image.shape[:2]
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) != 4:
                raise ValueError('Invalid polygon')
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                    raise ValueError('Invalid point')
                if not (0 <= point[0] <= w and 0 <= point[1] <= h): raise ValueError('Point outside image')
        args = build_items_from_ocr_res(ocr, image)
        if args[1]:
            result, labels = self.runner.infer_with_meta(image, *args)
        else:
            result = dict(name=self.runner.checkpoint.name, items=[], non_o_indices=[], kept_count=0, non_o_count=0, num_spans=0, stride=128, infer_elapsed_sec=0)
            labels = []
        result['ocr_meta'] = dict(endpoint='grpc', elapsed_sec=request.elapsed_seconds,
                                 returned_image_h=h, returned_image_w=w, raw_items=len(texts), kept_items=len(args[1]))
        overlay = draw_overlay(image, args[-1], labels, bgr_from_label_palette(list(self.runner.label2id)))
        ok, encoded = cv2.imencode('.jpg', overlay)
        if not ok: raise RuntimeError('Cannot encode visualization')
        return pb.LayoutResult(result_json=json.dumps(result, ensure_ascii=False).encode(), visualization=encoded.tobytes())


if __name__ == '__main__':
    serve(Service(), rpc.add_LayoutServiceServicer_to_server)
