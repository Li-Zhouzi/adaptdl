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

echo "[INFO] Sync attempt for dataset: $DATASET_NAME"

# Atomically move into final path if not already there
if mkdir "$FINAL_PATH" 2>/dev/null; then
  echo "[INFO] Acquired lock. Syncing from: $S3_BUCKET/$DATASET_NAME/"
  aws s3 sync "$S3_BUCKET/$DATASET_NAME/" "$FINAL_PATH/"
  
  # Check if the directory is non-empty
  if [ -z "$(ls -A "$FINAL_PATH")" ]; then
    echo "[ERROR] Dataset sync failed or returned empty. Check S3 path and credentials."
    rm -rf "$FINAL_PATH"  # Clean up partial sync
    exit 1
  fi

  touch "$READY_MARKER"
  echo "[INFO] Dataset ready at $FINAL_PATH"
else
  echo "[INFO] Another pod started first. Waiting for dataset to be ready..."

  # Wait until the READY_MARKER file exists
  while [ ! -f "$READY_MARKER" ]; do
    sleep 2
    echo "[INFO] Waiting for $READY_MARKER to appear..."
  done
  echo "[INFO] Detected ready dataset at $FINAL_PATH"
fi

echo "[DEBUG] Contents of $FINAL_PATH:"
ls -lh "$FINAL_PATH"

# Step 4: Start actual training
sleep 3
echo "[INFO] Starting training command: $@"
exec "$@"
