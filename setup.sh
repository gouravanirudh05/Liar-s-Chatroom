#!/usr/bin/env bash
# Environment for The Liar's Chatroom pipeline (runs on Kaggle GPU T4 x2).
# Downloads: Python packages from PyPI, and open-weight models from Hugging Face on first use by vLLM:
#   https://huggingface.co/Qwen/Qwen3-14B-AWQ   (Apache-2.0)
#   https://huggingface.co/Qwen/Qwen3-32B-AWQ   (Apache-2.0)
# Competition data: https://www.kaggle.com/competitions/the-liars-chatroom/data (place in Data/)
set -euo pipefail
pip install -q vllm pandas
pip install -q kaggle six   # only needed to push/fetch Kaggle kernels
python3 scripts/eval_rules.py  # sanity check: rule coverage and precision on train
