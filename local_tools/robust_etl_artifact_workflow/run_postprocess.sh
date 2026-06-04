#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "starting robust etl postprocessor..."
echo

if [ -f ".venv/bin/activate" ]; then
  . ".venv/bin/activate"
fi

streamlit run postprocessv8.py --server.maxUploadSize 10000
