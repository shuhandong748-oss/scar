#!/usr/bin/env bash
# TEST13 coordinator: dryload -> locked GPU export -> CPU features/eval/verify.
# Every stage is independently re-runnable.  Nothing here deletes data, kills
# processes, touches TRAIN/VAL caches or writes outside ROUND.
#
#   bash run_test13.sh dryload
#   bash run_test13.sh gpu        # takes the unified GPU lock
#   bash run_test13.sh cpu
#   bash run_test13.sh manifest
set -uo pipefail

PY=/root/miniconda3/envs/des4pos/bin/python
ROUND=/root/autodl-tmp/des4pos_paper_reimplementation/runs/corrected_kitti_test13_20261102
CODE=$ROUND/code
LOGS=$ROUND/logs
OUT=$ROUND/out
LOCK=/root/autodl-tmp/efficiency_audit_20261029/with_gpu_lock.sh
mkdir -p "$LOGS" "$OUT/state"

cmd=${1:-}
case "$cmd" in
dryload)
    echo "[dryload] $(date -Is)"
    cd "$CODE" || exit 1
    "$PY" "$CODE/test13_export.py" --mode dryload --device cuda --out-root "$OUT" \
        2>&1 | tee -a "$LOGS/dryload.log"
    exit "${PIPESTATUS[0]}"
    ;;
gpu)
    echo "[gpu] $(date -Is)"
    free_mb=$(df -Pm /root/autodl-tmp | awk 'NR==2{print $4}')
    echo "[gpu] free_mb=$free_mb"
    if [ "$free_mb" -lt 400 ]; then
        echo "[gpu] refusing: less than 400 MiB free on /root/autodl-tmp"
        exit 78
    fi
    if ps -eww -o args= | grep -E 'test13_export\.py' \
        | grep -vE 'grep|run_test13\.sh' >/dev/null; then
        echo "[gpu] refusing: another test13_export.py is running"
        exit 78
    fi
    EFF_GPU_LOCK_TIMEOUT="${EFF_GPU_LOCK_TIMEOUT:-21600}" \
        bash "$LOCK" bash -c '
            set -uo pipefail
            PY=/root/miniconda3/envs/des4pos/bin/python
            ROUND=/root/autodl-tmp/des4pos_paper_reimplementation/runs/corrected_kitti_test13_20261102
            cd "$ROUND/code" || exit 1
            echo "[inside] start $(date -Is) pid=$$"
            "$PY" test13_export.py --mode probe --device cuda --out-root "$ROUND/out"
            rc=$?; echo "[inside] probe rc=$rc"; [ $rc -ne 0 ] && exit $rc
            "$PY" test13_export.py --mode full --device cuda --out-root "$ROUND/out"
            rc=$?; echo "[inside] full rc=$rc"; exit $rc
        ' 2>&1 | tee -a "$LOGS/gpu.log"
    exit "${PIPESTATUS[0]}"
    ;;
cpu)
    echo "[cpu] $(date -Is)"
    cd "$CODE" || exit 1
    "$PY" test13_features.py --split test --out-root "$OUT" 2>&1 | tee "$LOGS/features.log"
    rc=${PIPESTATUS[0]}; echo "[cpu] features rc=$rc"; [ $rc -ne 0 ] && exit $rc
    "$PY" test13_eval.py --out-root "$OUT" --cross-check 2>&1 | tee "$LOGS/eval.log"
    rc=${PIPESTATUS[0]}; echo "[cpu] eval rc=$rc"; [ $rc -ne 0 ] && exit $rc
    "$PY" test13_verify.py 2>&1 | tee "$LOGS/verify.log"
    rc=${PIPESTATUS[0]}; echo "[cpu] verify rc=$rc"; exit $rc
    ;;
manifest)
    echo "[manifest] $(date -Is)"
    cd "$ROUND" || exit 1
    {
        find code out -type f \
            ! -name '*.tmp' ! -name 'SHA256SUMS*' ! -path 'out/state/ROUND_DONE*' \
            -print0 | sort -z | xargs -0 sha256sum
    } > "$ROUND/SHA256SUMS"
    wc -l "$ROUND/SHA256SUMS"
    echo "TEST13_DONE $(date -Is)" | tee "$ROUND/out/state/ROUND_DONE"
    exit 0
    ;;
*)
    echo "usage: bash run_test13.sh {dryload|gpu|cpu|manifest}" >&2
    exit 2
    ;;
esac
