#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv run --no-project --python 3.12 --with grpcio-tools==1.84.0 --with protobuf==7.36.2 python -m grpc_tools.protoc \
  -I proto --python_out=packages/protocol --grpc_python_out=packages/protocol proto/ngn_fba_ai_proto/ai.proto
