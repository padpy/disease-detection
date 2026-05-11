#!/usr/bin/env bash
# Set up the on-device segmentation model bundled with the Flutter app.
#
# Downloads the MobileSAM checkpoint (~40 MB) and exports the encoder and
# decoder to ONNX, dropping them in `assets/models/` under the names the
# Dart service (`local_segmentation_service.dart`) expects:
#
#   assets/models/sam3_efficient_encoder.onnx
#   assets/models/sam3_efficient_decoder.onnx
#
# MobileSAM is the SAM-family variant chosen for the mobile build because
# the encoder fits in ~40 MB and runs in a few hundred ms on a phone. The
# Dart service only depends on the standard SAM ONNX schema, so the
# filenames are a label — any compatible export (MobileSAM, EdgeSAM,
# EfficientSAM, SAM3-Efficient) works without touching the app.
#
# Usage:
#   ./tools/install_sam_model.sh
#
# Re-run any time you want to refresh the bundle (set REFRESH=1 to force
# a clean re-export).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSETS_DIR="${APP_DIR}/assets/models"
VENV_DIR="${SCRIPT_DIR}/.venv"
CHECKPOINT_DIR="${SCRIPT_DIR}/.cache"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/mobile_sam.pt"
CHECKPOINT_URL="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

ENCODER_OUT="${ASSETS_DIR}/sam3_efficient_encoder.onnx"
DECODER_OUT="${ASSETS_DIR}/sam3_efficient_decoder.onnx"
YOLO_OUT="${ASSETS_DIR}/yolo26_wheat_head.onnx"
YOLO_CHECKPOINT="${APP_DIR}/model/yolo_26_wheat_v1.0.0.pt"

REFRESH="${REFRESH:-0}"

log() { printf '\033[1;34m[sam-setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[sam-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "python3 not found in PATH"

mkdir -p "${ASSETS_DIR}" "${CHECKPOINT_DIR}"

if [[ "${REFRESH}" == "1" ]]; then
  log "REFRESH=1 — clearing previous outputs"
  rm -f "${ENCODER_OUT}" "${DECODER_OUT}"
fi

if [[ -f "${ENCODER_OUT}" && -f "${DECODER_OUT}" ]]; then
  log "ONNX models already present:"
  log "  ${ENCODER_OUT}"
  log "  ${DECODER_OUT}"
  log "Set REFRESH=1 to re-export."
  exit 0
fi

# 1. Python venv
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

log "Upgrading pip"
python -m pip install --quiet --upgrade pip wheel

log "Installing PyTorch + ONNX + MobileSAM + Ultralytics (this can take a few minutes the first time)"
python -m pip install --quiet \
  "torch>=2.1" \
  "torchvision>=0.16" \
  "onnx>=1.15" \
  "onnxruntime>=1.16" \
  "onnxslim>=0.1.34" \
  "timm>=0.6.13" \
  "ultralytics>=8.3.0" \
  "numpy<2"
# MobileSAM is shipped as a source package on GitHub, not on PyPI.
python -m pip install --quiet "git+https://github.com/ChaoningZhang/MobileSAM.git"

# 2. Checkpoint
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  log "Downloading MobileSAM checkpoint from ${CHECKPOINT_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar "${CHECKPOINT_URL}" -o "${CHECKPOINT_PATH}"
  else
    python -c "import urllib.request, sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "${CHECKPOINT_URL}" "${CHECKPOINT_PATH}"
  fi
fi
log "Checkpoint: ${CHECKPOINT_PATH} ($(du -h "${CHECKPOINT_PATH}" | cut -f1))"

# 3. Export
log "Exporting ONNX encoder + decoder → ${ASSETS_DIR}"
python "${SCRIPT_DIR}/export_sam_onnx.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --encoder-out "${ENCODER_OUT}" \
  --decoder-out "${DECODER_OUT}"

if [[ "${REFRESH}" == "1" ]]; then
  rm -f "${YOLO_OUT}"
fi
if [[ -f "${YOLO_OUT}" ]]; then
  log "YOLO26 ONNX already present: ${YOLO_OUT} ($(du -h "${YOLO_OUT}" | cut -f1))"
elif [[ -f "${YOLO_CHECKPOINT}" ]]; then
  log "Exporting YOLO26 wheat-head detector → ${YOLO_OUT}"
  python "${SCRIPT_DIR}/export_yolo_onnx.py" \
    --checkpoint "${YOLO_CHECKPOINT}" \
    --output "${YOLO_OUT}"
  log "YOLO26: ${YOLO_OUT} ($(du -h "${YOLO_OUT}" | cut -f1))"
else
  log "WARNING: ${YOLO_OUT} is missing and no checkpoint was found at"
  log "  ${YOLO_CHECKPOINT}"
  log "  Drop a single-class YOLO26 wheat-head .pt at that path (or the"
  log "  ONNX directly at ${YOLO_OUT}) before running the app."
fi

log ""
log "Done."
log "Encoder: ${ENCODER_OUT} ($(du -h "${ENCODER_OUT}" | cut -f1))"
log "Decoder: ${DECODER_OUT} ($(du -h "${DECODER_OUT}" | cut -f1))"
log ""
log "Next:"
log "  cd ${APP_DIR}"
log "  flutter clean && flutter pub get"
log "  (cd ios && pod install) # iOS only"
log "  flutter run --release"
