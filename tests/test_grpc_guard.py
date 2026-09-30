"""Run with a gRPC-enabled Python and PYTHONPATH=packages/protocol."""
from concurrent.futures import ThreadPoolExecutor
import grpc
from ngn_fba_ai_proto import ai_pb2 as pb, ai_pb2_grpc as rpc
from ngn_fba_ai_proto.server import Guard


class Service(rpc.CleanServiceServicer):
    def Health(self, request, context):
        return pb.HealthResponse(model='test')
    def Clean(self, request, context):
        if request.kind != pb.CleanRequest.TRACK:
            raise ValueError('Invalid kind')
        return pb.CleanResult(texts=request.texts)


server = grpc.server(ThreadPoolExecutor(max_workers=4), interceptors=[Guard('secret')])
rpc.add_CleanServiceServicer_to_server(Service(), server)
port = server.add_insecure_port('127.0.0.1:0')
server.start()
try:
    with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
        client = rpc.CleanServiceStub(channel)
        for method, request, metadata, expected in (
            (client.Health, pb.Empty(), (), grpc.StatusCode.UNAUTHENTICATED),
            (client.Clean, pb.CleanRequest(), (('x-api-key','secret'),), grpc.StatusCode.INVALID_ARGUMENT),
        ):
            try:
                method(request, metadata=metadata, timeout=2)
                raise AssertionError('Invalid request was accepted')
            except grpc.RpcError as error:
                assert error.code() == expected, error.code()
        assert client.Health(pb.Empty(), metadata=(('x-api-key','secret'),), timeout=2).model == 'test'
        response = client.Clean(pb.CleanRequest(kind=pb.CleanRequest.TRACK,texts=['a','','b']),metadata=(('x-api-key','secret'),),timeout=2)
        assert list(response.texts) == ['a','','b']
finally:
    server.stop(0).wait()
print('gRPC authentication, validation and batch alignment passed')
