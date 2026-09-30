import os
import grpc
from . import ai_pb2 as pb, ai_pb2_grpc as rpc
from .server import api_key

kind = os.environ['SERVICE']
stub = {'ocr': rpc.OcrServiceStub, 'layout': rpc.LayoutServiceStub, 'clean': rpc.CleanServiceStub}[kind]
with grpc.insecure_channel('localhost:50051') as channel:
    response = stub(channel).Health(pb.Empty(), metadata=(('x-api-key', api_key()),), timeout=5)
    assert response.model
