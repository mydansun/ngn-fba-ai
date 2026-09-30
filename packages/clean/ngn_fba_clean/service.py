import os
from pathlib import Path
import torch
from ngn_fba_ai_proto import ai_pb2 as pb, ai_pb2_grpc as rpc
from ngn_fba_ai_proto.server import serve
from .runtime import load_vocab, load_model_from_ckpt, Cleaner


class Service(rpc.CleanServiceServicer):
    def __init__(self):
        torch.set_num_threads(4)
        self.cleaners = {}
        for kind, name in ((pb.CleanRequest.TRACK, 'track'), (pb.CleanRequest.SKU, 'sku'), (pb.CleanRequest.RECIPIENT, 'recipient')):
            root = Path(os.environ['MODEL_DIR']) / name
            vocab = load_vocab(root / 'vocab.json')
            model, config = load_model_from_ckpt(root / 'model.ckpt', vocab, torch.device('cuda'))
            self.cleaners[kind] = Cleaner(model, vocab, config, torch.device('cuda'))

    def Health(self, request, context):
        return pb.HealthResponse(model=os.getenv('MODEL_VERSION', 'final-fit-v1'))

    def Clean(self, request, context):
        if request.kind not in self.cleaners or not 1 <= len(request.texts) <= 256 or any(len(t) > 512 for t in request.texts):
            raise ValueError('Invalid cleaner kind, batch or length')
        texts, _ = self.cleaners[request.kind].clean_batch_with_meta(list(request.texts), .5, 512, 256)
        return pb.CleanResult(texts=texts)


if __name__ == '__main__':
    serve(Service(), rpc.add_CleanServiceServicer_to_server)
