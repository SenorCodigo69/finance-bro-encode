#!/usr/bin/env bash
# backup.sh — Timestamped backup of critical untracked data
# Usage: ./scripts/backup.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_BASE="${PROJECT_ROOT}/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_BASE}/data_${TIMESTAMP}"
MAX_BACKUPS=5

echo "=== Finance Agent Backup ==="
echo "Backup target: ${BACKUP_DIR}"
echo ""

mkdir -p "${BACKUP_DIR}"

# Back up data/
DATA_DIR="${PROJECT_ROOT}/data"
if [ -d "${DATA_DIR}" ]; then
    echo "Backing up data/ ..."
    cp -R "${DATA_DIR}" "${BACKUP_DIR}/data"
else
    echo "WARNING: data/ not found, skipping"
fi

# Back up config/settings.yaml
SETTINGS="${PROJECT_ROOT}/config/settings.yaml"
if [ -f "${SETTINGS}" ]; then
    echo "Backing up config/settings.yaml ..."
    mkdir -p "${BACKUP_DIR}/config"
    cp "${SETTINGS}" "${BACKUP_DIR}/config/settings.yaml"
else
    echo "WARNING: config/settings.yaml not found, skipping"
fi

# Summary
echo ""
if command -v du &>/dev/null; then
    du -sh "${BACKUP_DIR}"/* 2>/dev/null | while read -r size path; do
        echo "  ${size}  $(basename "${path}")"
    done
    TOTAL=$(du -sh "${BACKUP_DIR}" | cut -f1)
    echo "Total: ${TOTAL}"
fi

# Rotate: keep last N backups
BACKUP_COUNT=$(ls -1d "${BACKUP_BASE}"/data_* 2>/dev/null | wc -l | tr -d ' ')
if [ "${BACKUP_COUNT}" -gt "${MAX_BACKUPS}" ]; then
    REMOVE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    echo ""
    echo "Rotating (keeping last ${MAX_BACKUPS}, removing ${REMOVE_COUNT}) ..."
    ls -1d "${BACKUP_BASE}"/data_* | head -n "${REMOVE_COUNT}" | while read -r old; do
        echo "  Removing $(basename "${old}")"
        rm -rf "${old}"
    done
fi

echo ""
echo "Done: ${BACKUP_DIR}"
