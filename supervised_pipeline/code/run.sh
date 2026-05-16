#!/usr/bin/env bash
# SimCSE training pipeline on A10 (CUDA, 24GB VRAM, bf16/fp16 supported)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "(nvidia-smi not found)"
echo

echo "=== Python / Torch info ==="
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
echo

echo "=== Smoke test (500 verdicts, ~3-5 min) ==="
python train_simcse.py --limit 500 --epochs 1
echo

echo "=== Full training (~30-60 min on A10) ==="
python train_simcse.py
echo

echo "=== Done. Outputs in: ./outputs/ ==="
ls -lh outputs/
