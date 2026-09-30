"""Bounded, authenticated unary GPU services."""
from concurrent.futures import ThreadPoolExecutor
import hmac
import logging
import os
from pathlib import Path
import signal
import threading
import grpc

OPTIONS = [('grpc.max_receive_message_length', 50 * 1024 * 1024),
           ('grpc.max_send_message_length', 50 * 1024 * 1024)]


def api_key():
    key = Path(os.environ['AI_API_KEY_FILE']).read_text().strip()
    if len(key) < 32:
        raise ValueError('AI API key must have at least 32 characters')
    return key


class Guard(grpc.ServerInterceptor):
    def __init__(self, key):
        self.key = key.encode()
        # ponytail: serialize each GPU engine; use batching if measured traffic needs it.
        self.lock = threading.Lock()

    def intercept_service(self, continuation, details):
        handler = continuation(details)
        if handler is None:
            return None
        def call(request, context):
            supplied = dict(context.invocation_metadata()).get('x-api-key', '').encode()
            if not hmac.compare_digest(supplied, self.key):
                context.abort(grpc.StatusCode.UNAUTHENTICATED, 'Invalid API key')
            if details.method.endswith('/Health'):
                return handler.unary_unary(request, context)
            remaining = context.time_remaining()
            if not self.lock.acquire(timeout=max(0, min(120, remaining if remaining is not None else 120))):
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, 'Inference queue timeout')
            try:
                if not context.is_active():
                    context.abort(grpc.StatusCode.CANCELLED, 'Request cancelled')
                try:
                    return handler.unary_unary(request, context)
                except ValueError:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Invalid inference input')
                except Exception:
                    logging.exception('Inference failed')
                    context.abort(grpc.StatusCode.INTERNAL, 'Inference failed')
            finally:
                self.lock.release()
        return grpc.unary_unary_rpc_method_handler(call, request_deserializer=handler.request_deserializer,
                                                   response_serializer=handler.response_serializer)


def serve(service, register):
    server = grpc.server(ThreadPoolExecutor(max_workers=4), interceptors=[Guard(api_key())],
                         options=OPTIONS, maximum_concurrent_rpcs=8)
    register(service, server)
    if not server.add_insecure_port(os.getenv('GRPC_BIND', '0.0.0.0:50051')):
        raise RuntimeError('Cannot bind gRPC port')
    signal.signal(signal.SIGTERM, lambda *_: server.stop(30))
    server.start()
    server.wait_for_termination()


def decode_image(data):
    import io
    import cv2
    import numpy as np
    from PIL import Image
    if not data or len(data) > 20 * 1024 * 1024:
        raise ValueError('Invalid image length')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > 25_000_000 or im.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('Invalid image dimensions or format')
            im.verify()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Invalid image') from error
    if image is None:
        raise ValueError('Invalid image')
    return image
