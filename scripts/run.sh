#!/bin/bash

set -euo pipefail

# require argument
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <label>"
  exit 1
fi

LABEL="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

rm -rf "$PROJECT_ROOT/sample_results/"
make clean
make
python "$SCRIPT_DIR/trace_exec_training_list.py" --trace_dir "$PROJECT_ROOT/sample_traces/" --results_dir "$PROJECT_ROOT/sample_results/"
python "$SCRIPT_DIR/append_result.py" "$LABEL"
