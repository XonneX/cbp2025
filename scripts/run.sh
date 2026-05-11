#!/bin/bash

set -euo pipefail

print_help() {
  cat <<EOF
Usage: $0 [label]

Runs the full pipeline and optionally appends results.

Arguments:
  label        Optional label to prepend when appending results

Options:
  -h, --help   Show this help message and exit

Examples:
  $0 Always return false
  $0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

LABEL="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

rm -rf "$PROJECT_ROOT/sample_results/"

if [ -z "$LABEL" ]; then
  make clean
fi

make
python "$SCRIPT_DIR/trace_exec_training_list.py" --trace_dir "$PROJECT_ROOT/sample_traces/" --results_dir "$PROJECT_ROOT/sample_results/"

if [ -n "$LABEL" ]; then
  python "$SCRIPT_DIR/append_result.py" "$LABEL"
fi
