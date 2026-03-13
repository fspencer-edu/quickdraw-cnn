#!/usr/bin/env bash
set -e

MODEL_SOURCE="/mnt/ml/models/quickdraw-cnn/v1/model.keras"
STAGING_TARGET_DIR="/mnt/ml/registry/staging/quickdraw-cnn"

mkdir -p "${STAGING_TARGET_DIR}"
cp "${MODEL_SOURCE}" "${STAGING_TARGET_DIR}/model.keras"

echo "Copied ${MODEL_SOURCE} -> ${STAGING_TARGET_DIR}/model.keras"