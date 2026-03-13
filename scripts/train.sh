#!/usr/bin/env bash
set -e
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH=src
python -m quickdraw_cnn.train