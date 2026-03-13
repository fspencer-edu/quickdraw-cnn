#!/usr/bin/env bash
set -e

source .venv/bin/activate
export PYTHONPATH=src
python -m quickdraw_cnn.evaluate