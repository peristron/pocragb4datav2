#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "starting local csv anonymizer and chunker..."
echo

if [ -f ".venv/bin/activate" ]; then
  . ".venv/bin/activate"
fi

streamlit run csv_anonymizer_chunker.py --server.maxUploadSize 10000
