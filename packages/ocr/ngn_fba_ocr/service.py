import json
import time
from ngn_fba_ai_proto import ai_pb2 as pb, ai_pb2_grpc as rpc
from ngn_fba_ai_proto.server import serve, decode_image
from .runtime import create_ocr, extract_ocr_res_and_output, bgr_ndarray_to_jpeg_bytes


class Service(rpc.OcrServiceServicer):
    def __init__(self):
        self.engine = create_ocr(device='gpu:0')

    def Health(self, request, context):
        return pb.HealthResponse(model='PP-OCRv5-server')

    def Recognize(self, request, context):
        image = decode_image(request.image)
        started = time.perf_counter()
        enabled = request.unwarping if request.HasField('unwarping') else True
        results = list(self.engine.predict(image, use_doc_unwarping=enabled))
        result, processed = extract_ocr_res_and_output(results)
        # Keep only the arrays consumed by LayoutLM, avoiding preprocessing image arrays.
        ocr = {key: result[key] for key in ('rec_texts', 'rec_polys', 'rec_scores')}
        return pb.OcrResult(image=bgr_ndarray_to_jpeg_bytes(processed),
                            ocr_json=json.dumps(ocr, ensure_ascii=False).encode(),
                            elapsed_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    serve(Service(), rpc.add_OcrServiceServicer_to_server)
