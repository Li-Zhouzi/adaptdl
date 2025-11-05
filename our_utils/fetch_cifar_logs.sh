#!/usr/bin/env bash
set -euo pipefail

# Usage: ./fetch_cifar_logs.sh 0 1 2 3
# Optionally set NAMESPACE to target a specific kube namespace.
#   Example: NAMESPACE=default ./fetch_cifar_logs.sh 0 1 2

POD_PREFIX="cifar10-35-e0691978-7ad3-41a8-8a0e-ee032af23ba9-4-"
OUT_DIR="./experiment_results/1031-restart-test"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <n1> [n2 n3 ...]"
  exit 1
fi

mkdir -p "${OUT_DIR}"

for n in "$@"; do
  pod="${POD_PREFIX}${n}"
  out_file="${OUT_DIR}/restart4-pod${n}"
  echo "Fetching logs for pod: ${pod} -> ${out_file}"
  if [[ -n "${NAMESPACE:-}" ]]; then
    kubectl logs -n "${NAMESPACE}" "${pod}" > "${out_file}" 2>&1 || echo "Failed: ${pod}"
  else
    kubectl logs "${pod}" > "${out_file}" 2>&1 || echo "Failed: ${pod}"
  fi
done

echo "Done."


