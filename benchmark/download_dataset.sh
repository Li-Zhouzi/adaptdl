#!/bin/bash
set -e

JOB_NAME="$1"
shift 

# Load dataset mapping from JSON
DATASET_NAME=$(python3 -c "import json; print(json.load(open('/root/dataset_map.json'))['$JOB_NAME'])")

S3_BUCKET="s3://adaptdl-datasets"
DATA_ROOT="/mnt"
FINAL_PATH="$DATA_ROOT/$DATASET_NAME"
READY_MARKER="$FINAL_PATH/.ready"

# Unique temporary path for this pod
TMP_PATH="$DATA_ROOT/tmp-${JOB_NAME}-$$"

if [ -f "$READY_MARKER" ]; then
  echo "[INFO] Dataset already present at $FINAL_PATH"
else
  echo "[INFO] Downloading dataset to temp path $TMP_PATH..."
  mkdir -p "$TMP_PATH"
  echo "[INFO] Syncing from: $S3_BUCKET/$DATASET_NAME/"
  aws s3 sync "$S3_BUCKET/$DATASET_NAME/" "$TMP_PATH/"
  if [ -z "$(ls -A "$TMP_PATH")" ]; then
  echo "[ERROR] Dataset sync failed or returned empty. Check S3 path and credentials."
  exit 1
  fi

  echo "[INFO] Sync complete. Attempting atomic move..."

  # Atomically move into final path if not already there
  if mkdir "$FINAL_PATH" 2>/dev/null; then
    mv "$TMP_PATH"/* "$FINAL_PATH"/
    touch "$READY_MARKER"
    echo "[INFO] Dataset ready at $FINAL_PATH"
  else
    echo "[INFO] Another pod finished first. Cleaning up..."
    rm -rf "$TMP_PATH"
  fi
fi


echo "[DEBUG] contents of $FINAL_PATH:"
ls -lh "$FINAL_PATH"
# Step 4: Start actual training
sleep 3
echo "[INFO] Starting training command: $@"
exec "$@"
