#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${SCRIPT_DIR}"

MODEL_FILES=(
  xgb_rul.joblib
  MODEL_CARD.md
)

echo "Syncing serving files from ${ROOT} -> ${DEST}"

rm -rf "${DEST}/api" "${DEST}/models"
mkdir -p "${DEST}/models"

cp "${ROOT}/requirements.txt" "${DEST}/requirements.txt"
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "${ROOT}/api/" "${DEST}/api/"

for file in "${MODEL_FILES[@]}"; do
  src="${ROOT}/models/${file}"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${DEST}/models/${file}"
  else
    echo "WARNING: missing artifact ${src}" >&2
  fi
done

echo "Done. Staged files:"
find "${DEST}" -maxdepth 2 -type f | sort
