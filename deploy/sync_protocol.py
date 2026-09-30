"""Copy generated client bindings into scan's independent Docker build context."""
import argparse
import hashlib
from pathlib import Path
import shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('scan_backend', type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
dest=a.scan_backend/'src/ngn_fba_ai_proto';dest.mkdir(parents=True,exist_ok=True)
for name in ('__init__.py','ai_pb2.py','ai_pb2_grpc.py'):
    shutil.copyfile(root/'packages/protocol/ngn_fba_ai_proto'/name,dest/name)
sha=hashlib.sha256((root/'proto/ngn_fba_ai_proto/ai.proto').read_bytes()).hexdigest()
(dest/'SOURCE.txt').write_text('Generated from mydansun/ngn-fba-ai proto/ngn_fba_ai_proto/ai.proto\nSHA256 '+sha+'\nRegenerate with deploy/sync_protocol.py in the AI repository.\n')
