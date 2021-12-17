#!/bin/bash

set -e

cd "$(dirname "$0")"
PROTO_DIR=wtil/api

for item in $(ls -d $PROTO_DIR/*.proto | xargs dirname | sort | uniq);
do
    echo "processing dir: $item"
    python -m grpc_tools.protoc \
        -I. \
        --python_out=. \
        $item/*.proto
        # --grpc_python_out=. \
        # --go_out=paths=source_relative:. \
        # --go-grpc_out=paths=source_relative:. \
done

for item in $(find $PROTO_DIR -name "*_pb2_grpc\.py");
do
    echo "processing file: $item"
    sed -i 's/^from \([^ ]\+\) import \([^ ]\+\) as \([^ ]\+\)$/from . import \2 as \3/' $item
done
