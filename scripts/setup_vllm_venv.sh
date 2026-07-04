#!/usr/bin/env bash
# setup_vllm_venv.sh — standalone vLLM venv for the CoT elicitation run.
#
# vLLM (>=0.10, torch>=2.7, transformers<4.56) is UNSATISFIABLE against the project's
# torch>=2.6,<2.7 + transformers>=5.0, so it cannot share the project .venv. This builds
# a SEPARATE venv (.venv-vllm) with only vllm + python-dotenv. The CoT scripts import the
# repo's pure-Python helpers via PYTHONPATH=src (no project deps needed for scoring).
#
# Usage:  bash scripts/setup_vllm_venv.sh
# Then:   PYTHONPATH=src .venv-vllm/bin/python scripts/phase1/task0_vllm_probe.py
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv-vllm"
echo "=== vLLM venv setup in $VENV ==="

# CRITICAL: build + install from $HOME, NOT the project dir, so uv does NOT apply the
# project's [tool.uv.index] cu124 pin (which lacks the torch 2.7.1 vLLM needs). vLLM's own
# PyPI deps fetch a compatible CUDA torch. The venv is referenced by absolute path.
cd "$HOME"
rm -rf "$VENV"  # clear any partial venv from a failed run
uv venv "$VENV" --python 3.12

# The box driver is 570.x / CUDA 12.8. vLLM 0.23.0 is compiled against CUDA 13 (its _C ext
# needs libcudart.so.13), incompatible with this driver. vLLM 0.11.0 pins torch==2.8.0 and
# ships CUDA-12.8 wheels (libcudart.so.12) -> matches the driver. It also contains PR #20905
# (Devstral tekken.json v13 fix, merged ~0.10), so it covers all 5 models. Pin it explicitly.
# vLLM 0.11.0 declares transformers>=4.55.2 (no upper bound), so uv grabs transformers 5.x,
# whose Qwen2Tokenizer dropped `all_special_tokens_extended` that vLLM 0.11 calls -> crash.
# Cap transformers to the 4.x line vLLM 0.11 was built against.
uv pip install --python "$VENV/bin/python" \
    --torch-backend cu128 \
    "vllm==0.11.0" "transformers>=4.55.2,<5" python-dotenv

echo ""
echo "vLLM venv ready. Smoke check:"
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
