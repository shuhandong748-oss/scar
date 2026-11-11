#!/usr/bin/env bash
# Serial driver for the TRAIN/VAL-only round: NPM3D first, then S3DIS.
# Stops after both.  Does NOT arm or read TEST, does NOT start efficiency,
# does NOT shut the instance down, does not touch A/B.
set -uo pipefail

ROOT=/root/autodl-tmp/d4p_trainval_20261104/C
CODE=$ROOT/code
LOGS=$ROOT/logs
mkdir -p "$LOGS"
PY=""
for cand in /root/miniconda3/envs/des4pos/bin/python python3 python; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import torch, torch_geometric, numpy" >/dev/null 2>&1; then
        PY=$cand; break
    fi
done
[ -n "$PY" ] || { echo "[both] no python with torch+torch_geometric" | tee -a "$LOGS/both.log"; exit 5; }
echo "[both] start $(date -Is) python=$PY" | tee -a "$LOGS/both.log"

for DOM in npm3d s3dis; do
    echo "[both] $DOM begin $(date -Is)" | tee -a "$LOGS/both.log"
    bash "$CODE/run_domain_trainval.sh" "$DOM"
    rc=$?
    echo "[both] $DOM rc=$rc $(date -Is)" | tee -a "$LOGS/both.log"
    if [ $rc -ne 0 ]; then
        echo "[both] stop after $DOM (rc=$rc); S3DIS not started" | tee -a "$LOGS/both.log"
        "$PY" - "$ROOT/HANDOFF.json" "STOPPED_AT_$DOM" <<'PYEOF'
import json, sys, time
json.dump({"status": sys.argv[2], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "test_read": False, "test_armed": False,
           "efficiency_started": False, "auto_shutdown": False,
           "note": "chain stopped on failure; manual review required"},
          open(sys.argv[1], "w"), indent=2)
PYEOF
        exit $rc
    fi
done

# ------------------------------------------------------------- handoff
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
    > "$LOGS/compute_apps_after.txt" 2>&1
ps -eo pid,etime,args --sort=-etime | grep -E "d4p_|des4pos" | grep -v grep \
    > "$LOGS/procs_after.txt" 2>&1 || true
if flock -n /root/autodl-tmp/efficiency_audit_20261029/GPU.lock true 2>/dev/null; then
    LOCK_FREE=true
else
    LOCK_FREE=false
fi

"$PY" - "$ROOT" "$ROOT/HANDOFF.json" "$LOCK_FREE" <<'PYEOF'
import hashlib, json, os, sys, time
root, out_path, lock_free = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
out = {"status": "COMPLETE_TRAINVAL", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "domains": {}, "test_read": False, "test_armed": False,
       "efficiency_started": False, "auto_shutdown": False,
       "gpu_lock_free": lock_free,
       "next": "TEST remains unread and unarmed; another round must hash-freeze "
               "this delivery before any TEST export"}
for dom in ("npm3d", "s3dis"):
    d = os.path.join(root, dom)
    rec = {"run_dir": d}
    for name in ("DOMAIN_STATUS.json", "FREEZE_LIST.json", "NO_TEST_POLICY.json"):
        p = os.path.join(d, name)
        rec[name] = json.load(open(p)) if os.path.isfile(p) else None
    m = os.path.join(d, "test", "TEST_READ_ONCE.json")
    rec["test_marker_sha256"] = (hashlib.sha256(open(m, "rb").read()).hexdigest()
                                 if os.path.isfile(m) else None)
    rec["tar_sha256"] = (open(os.path.join(d, "delivery", "TAR_SHA256.txt"))
                         .read().split()[0]
                         if os.path.isfile(os.path.join(d, "delivery",
                                                        "TAR_SHA256.txt")) else None)
    rec["sha_manifest"] = os.path.join(d, "SHA256SUMS")
    out["domains"][dom] = rec
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out, indent=2)[:2000])
PYEOF

echo "[both] handoff written $(date -Is); no TEST, no efficiency, no shutdown" \
    | tee -a "$LOGS/both.log"
