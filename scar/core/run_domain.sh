#!/usr/bin/env bash
# Corrected Des4Pos formal-SCAR chain, one domain at a time, serial.
#   stage A  identity / lock / disk / gate
#   stage B  GPU-locked VAL probe (2 queries), VAL export, TRAIN export
#   stage C  CPU: independent export verification, canonical-23 features
#   stage D  CPU: three formal heads (s0 main), VAL evaluation + verification
#   stage E  FREEZE_LIST (before any TEST), then arm TEST
#   stage F  one shared TEST export, features, evaluation, verification
#   stage G  package + SHA manifest + domain status
# Never: TEST before freeze, a second TEST export, retraining to chase numbers,
# efficiency measurement, shutdown, or touching A/B.
set -uo pipefail

DOM=${1:?usage: run_domain.sh <npm3d|s3dis>}
case "$DOM" in npm3d|s3dis) ;; *) echo "unknown domain $DOM" >&2; exit 2;; esac

ROOT=/root/autodl-tmp/d4p_trainval_20261104/C
RUN=$ROOT/$DOM
CODE=$ROOT/code
LOGS=$RUN/logs
PROTO=$ROOT/records/PROTOCOL_FREEZE.json
GATE=$ROOT/records/GATE_BINDING.json
LOCK=/root/autodl-tmp/efficiency_audit_20261029/with_gpu_lock.sh
EXPECT_HOST=autodl-container-<CONTAINER_ID>
EXPECT_GPU=GPU-<GPU_UUID>
MIN_FREE_BYTES=$((3 * 1024 * 1024 * 1024))

mkdir -p "$RUN" "$LOGS" "$RUN/export" "$RUN/features" "$RUN/heads" \
         "$RUN/verify" "$RUN/eval_val" "$RUN/probe" "$RUN/delivery"
echo "[main] $DOM start $(date -Is)" | tee "$LOGS/main.log"

PY=""
for cand in /root/miniconda3/envs/des4pos/bin/python python3 python; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c "import torch, torch_geometric, numpy" >/dev/null 2>&1; then
        PY=$cand; break
    fi
done
[ -n "$PY" ] || { echo "no python with torch+torch_geometric" | tee -a "$LOGS/main.log"; exit 5; }
"$PY" -c "import torch; print('python', __import__('sys').version.split()[0], 'torch', torch.__version__)" \
    | tee -a "$LOGS/main.log"

status_done() {
    local test_read=false
    [ -f "$RUN/test/TEST_READ_ONCE.json" ] && test_read=true
    "$PY" - "$RUN/DOMAIN_STATUS.json" "$DOM" "$1" "$2" "$test_read" <<'PYEOF'
import json, os, sys, time
path, dom, status, stage, test_read = sys.argv[1:6]
rec = {"domain": dom, "status": status, "stage": stage,
       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "test_access": test_read == "true",
       "efficiency_started": False, "auto_shutdown": False}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f, indent=2); f.write("\n")
os.replace(tmp, path)
PYEOF
}

run_step() {
    local name=$1; shift
    echo "[main] START $name $(date -Is)" | tee -a "$LOGS/main.log"
    local t0=$SECONDS
    "$@" > "$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[main] END   $name rc=$rc $((SECONDS-t0))s" | tee -a "$LOGS/main.log"
    if [ $rc -ne 0 ]; then
        echo "[main] STOP at $name; tail of $LOGS/$name.log:" | tee -a "$LOGS/main.log"
        tail -n 40 "$LOGS/$name.log" | tee -a "$LOGS/main.log"
        status_done FAILED "$name"
        exit $rc
    fi
}

gpu_step() {
    local name=$1; shift
    echo "[main] START $name (GPU lock) $(date -Is)" | tee -a "$LOGS/main.log"
    local t0=$SECONDS
    EFF_GPU_LOCK_TIMEOUT=1800 "$LOCK" "$@" > "$LOGS/$name.log" 2>&1
    local rc=$?
    echo "[main] END   $name rc=$rc $((SECONDS-t0))s" | tee -a "$LOGS/main.log"
    if [ $rc -ne 0 ]; then
        echo "[main] STOP at $name (lock/busy refusal or failure); tail:" | tee -a "$LOGS/main.log"
        tail -n 40 "$LOGS/$name.log" | tee -a "$LOGS/main.log"
        status_done FAILED "$name"
        exit $rc
    fi
}

# ---------------------------------------------------------------- stage A
{
    echo "hostname=$(hostname)"; echo "date=$(date -Is)"; id
    nvidia-smi -L; nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader
    df -B1 /root/autodl-tmp /
    free -m
} > "$LOGS/identity.txt" 2>&1
[ "$(hostname)" = "$EXPECT_HOST" ] || { echo "hostname mismatch" | tee -a "$LOGS/main.log"; status_done BLOCKED identity; exit 6; }
nvidia-smi -L | grep -q "$EXPECT_GPU" || { echo "GPU mismatch" | tee -a "$LOGS/main.log"; status_done BLOCKED gpu; exit 7; }
[ -f "$LOCK" ] || { echo "lock wrapper missing" | tee -a "$LOGS/main.log"; status_done BLOCKED lock; exit 8; }
[ -f "$GATE" ] && [ -f "$PROTO" ] || { echo "gate/protocol record missing" | tee -a "$LOGS/main.log"; status_done BLOCKED binding; exit 9; }
"$PY" - "$GATE" <<'PYEOF' | tee -a "$LOGS/main.log"
import json, sys
g = json.load(open(sys.argv[1]))
assert g.get("all_pass") is True, "gate not PASS"
print("gate PASS; gate sha", __import__("hashlib").sha256(open(sys.argv[1], "rb").read()).hexdigest())
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { status_done BLOCKED gate; exit 10; }
PROTO_SHA=$(sha256sum "$PROTO" | awk '{print $1}')
FREE=$(df -B1 --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
echo "[main] protocol $PROTO_SHA free $FREE" | tee -a "$LOGS/main.log"
[ "$FREE" -ge "$MIN_FREE_BYTES" ] || { echo "disk below 3 GiB budget" | tee -a "$LOGS/main.log"; status_done BLOCKED disk; exit 11; }
BUSY=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | \
       tr -d ' MiB' | awk '{s+=$1} END {print s+0}')
echo "[main] GPU compute memory in use: ${BUSY} MiB (lock gate enforces >=200 MiB)" \
    | tee -a "$LOGS/main.log"
"$PY" - "$RUN/SPACE_BUDGET.json" "$DOM" "$FREE" <<'PYEOF' | tee -a "$LOGS/main.log"
import json, sys, time
path, dom, free = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = {"npm3d": {"train": 120, "val": 128, "test": 116},
     "s3dis": {"train": 1084, "val": 544, "test": 544}}[dom]
d = 768
est = {}
for s, q in n.items():
    cand = q * 25 * 220                       # ids/scores/offsets/pred_w, compressed
    enc = (q + 8000) * d * 4                  # text + cell embeddings, upper bound
    feat = q * 25 * 23 * 4
    est[s] = dict(candidates_MiB=round(cand / 2**20, 1),
                  enc_MiB=round(enc / 2**20, 1),
                  features_MiB=round(feat / 2**20, 1),
                  total_MiB=round((cand + enc + feat) / 2**20, 1))
total = sum(v["total_MiB"] for v in est.values()) + 20  # heads/logs
rec = dict(domain=dom, free_bytes=free, estimate_MiB=est,
           total_with_packaging_MiB=round(total * 2 + 50, 1),
           budget_bytes=3 * 2**30, decision="proceed",
           note="no historical evidence is deleted; if free space fell below "
                "3 GiB the chain stops before exporting")
json.dump(rec, open(path, "w"), indent=2)
print("space budget", json.dumps(rec)[:400])
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { status_done BLOCKED space_budget; exit 13; }

cat > "$RUN/START_BINDING.json" <<EOF
{"domain": "$DOM", "hostname": "$(hostname)", "gpu_uuid": "$EXPECT_GPU",
 "protocol_freeze": "$PROTO", "protocol_sha256": "$PROTO_SHA",
 "gate_binding": "$GATE", "python": "$PY",
 "started": "$(date -Is)", "policy": "serial, GPU-locked exports, freeze before TEST"}
EOF

# ---------------------------------------------------------------- stage B
if [ -f "$RUN/probe/val2/EXPORT_META.json" ]; then
    echo "[main] probe exists, skip" | tee -a "$LOGS/main.log"
else
    gpu_step probe_val2 "$PY" "$CODE/d4p_export.py" --domain "$DOM" --split val \
        --out_dir "$RUN/probe/val2" --protocol_sha "$PROTO_SHA" --limit_queries 2
fi
if [ -f "$RUN/export/val/EXPORT_META.json" ]; then
    echo "[main] VAL export exists, skip" | tee -a "$LOGS/main.log"
else
    gpu_step export_val "$PY" "$CODE/d4p_export.py" --domain "$DOM" --split val \
        --out_dir "$RUN/export/val" --protocol_sha "$PROTO_SHA"
fi
if [ -f "$RUN/export/train/EXPORT_META.json" ]; then
    echo "[main] TRAIN export exists, skip" | tee -a "$LOGS/main.log"
else
    gpu_step export_train "$PY" "$CODE/d4p_export.py" --domain "$DOM" --split train \
        --out_dir "$RUN/export/train" --protocol_sha "$PROTO_SHA"
fi

# ---------------------------------------------------------------- stage C
run_step verify_export_val "$PY" "$CODE/d4p_verify_export.py" --domain "$DOM" \
    --split val --export_dir "$RUN/export/val" --probe_dir "$RUN/probe/val2" \
    --out "$RUN/verify/VERIFY_EXPORT_val.json"
run_step verify_export_train "$PY" "$CODE/d4p_verify_export.py" --domain "$DOM" \
    --split train --export_dir "$RUN/export/train" \
    --out "$RUN/verify/VERIFY_EXPORT_train.json"
if [ -f "$RUN/features/train.npz" ] && [ -f "$RUN/features/train.npz.meta.json" ]; then
    echo "[main] TRAIN features exist, skip" | tee -a "$LOGS/main.log"
else
    run_step features_train "$PY" "$CODE/d4p_features.py" --domain "$DOM" --split train \
        --export_dir "$RUN/export/train" --out "$RUN/features/train.npz" \
        --protocol_sha "$PROTO_SHA"
fi
if [ -f "$RUN/features/val.npz" ] && [ -f "$RUN/features/val.npz.meta.json" ]; then
    echo "[main] VAL features exist, skip" | tee -a "$LOGS/main.log"
else
    run_step features_val "$PY" "$CODE/d4p_features.py" --domain "$DOM" --split val \
        --export_dir "$RUN/export/val" --out "$RUN/features/val.npz" \
        --protocol_sha "$PROTO_SHA"
fi

# ---------------------------------------------------------------- stage D
if [ -f "$RUN/heads/SELECTION_FORMAL.json" ]; then
    echo "[main] heads exist, skip training" | tee -a "$LOGS/main.log"
else
    run_step heads "$PY" "$CODE/d4p_train_head.py" --domain "$DOM" \
        --train_npz "$RUN/features/train.npz" --val_npz "$RUN/features/val.npz" \
        --out_dir "$RUN/heads" --protocol_sha "$PROTO_SHA"
fi
if [ -f "$RUN/eval_val/metrics.json" ]; then
    echo "[main] VAL evaluation exists, skip" | tee -a "$LOGS/main.log"
else
    run_step eval_val "$PY" "$CODE/d4p_eval.py" --domain "$DOM" --split val \
        --export_dir "$RUN/export/val" --features "$RUN/features/val.npz" \
        --heads_dir "$RUN/heads" --out_dir "$RUN/eval_val" --protocol_sha "$PROTO_SHA"
fi
run_step verify_eval_val "$PY" "$CODE/d4p_verify_eval.py" --domain "$DOM" \
    --split val --export_dir "$RUN/export/val" --features "$RUN/features/val.npz" \
    --heads_dir "$RUN/heads" --eval_dir "$RUN/eval_val" \
    --protocol_sha "$PROTO_SHA" --out "$RUN/verify/VERIFY_EVAL_val.json"

# ---------------------------------------------------------------- stage E
if [ -f "$RUN/FREEZE_LIST.json" ]; then
    echo "[main] freeze list exists, reuse" | tee -a "$LOGS/main.log"
else
    run_step freeze "$PY" "$CODE/d4p_freeze.py" --domain "$DOM" \
        --run_dir "$RUN" --protocol_sha "$PROTO_SHA"
fi
FREEZE_SHA=$(awk '{print $1}' "$RUN/FREEZE_LIST.sha256")
echo "[main] freeze $FREEZE_SHA" | tee -a "$LOGS/main.log"
if [ -f "$RUN/test/TEST_READ_ONCE.json" ]; then
    echo "[main] TEST already armed/read; refusing a second read" | tee -a "$LOGS/main.log"
    status_done FAILED test_gate; exit 12
fi
run_step arm_test "$PY" "$CODE/d4p_arm_test.py" --domain "$DOM" \
    --run_dir "$RUN" --protocol_sha "$PROTO_SHA" --freeze_sha "$FREEZE_SHA"

# ---------------------------------------------------------------- stage F
gpu_step export_test "$PY" "$CODE/d4p_export.py" --domain "$DOM" --split test \
    --out_dir "$RUN/test" --protocol_sha "$PROTO_SHA" --freeze_sha "$FREEZE_SHA"
run_step verify_export_test "$PY" "$CODE/d4p_verify_export.py" --domain "$DOM" \
    --split test --export_dir "$RUN/test" --out "$RUN/verify/VERIFY_EXPORT_test.json"
if [ -f "$RUN/features/test.npz" ]; then
    echo "[main] TEST features exist, skip" | tee -a "$LOGS/main.log"
else
    run_step features_test "$PY" "$CODE/d4p_features.py" --domain "$DOM" --split test \
        --export_dir "$RUN/test" --out "$RUN/features/test.npz" \
        --protocol_sha "$PROTO_SHA" --freeze_sha "$FREEZE_SHA"
fi
run_step eval_test "$PY" "$CODE/d4p_eval.py" --domain "$DOM" --split test \
    --export_dir "$RUN/test" --features "$RUN/features/test.npz" \
    --heads_dir "$RUN/heads" --out_dir "$RUN/eval_test" --protocol_sha "$PROTO_SHA" \
    --freeze_sha "$FREEZE_SHA" --allow_test
run_step verify_eval_test "$PY" "$CODE/d4p_verify_eval.py" --domain "$DOM" \
    --split test --export_dir "$RUN/test" --features "$RUN/features/test.npz" \
    --heads_dir "$RUN/heads" --eval_dir "$RUN/eval_test" \
    --protocol_sha "$PROTO_SHA" --out "$RUN/verify/VERIFY_EVAL_test.json"

# ---------------------------------------------------------------- stage G
run_step code_snapshot bash -c "
    set -e
    rm -rf $RUN/code_snapshot
    mkdir -p $RUN/code_snapshot/adapt
    cp $CODE/*.py $CODE/*.sh $RUN/code_snapshot/
    cp $ROOT/records/*.json $RUN/code_snapshot/ 2>/dev/null || true
    cp /root/autodl-tmp/des4pos_corrected_C_20261102/code/adapt/*.py \
        $RUN/code_snapshot/adapt/ 2>/dev/null || true
    find /root/autodl-tmp/des4pos_corrected_C_20261102/code/core -maxdepth 1 \
        -iname '*manifest*' -exec cp {} $RUN/code_snapshot/ \\; 2>/dev/null || true
"
run_step report "$PY" "$CODE/d4p_report.py" --domain "$DOM" --run_dir "$RUN" \
    --protocol_sha "$PROTO_SHA"
run_step package bash -c "
    set -e
    cd $ROOT
    rm -f $RUN/delivery/${DOM}_corrected_scar.tar
    tar -cf $RUN/delivery/${DOM}_corrected_scar.tar $DOM
    sha256sum $RUN/delivery/${DOM}_corrected_scar.tar > $RUN/delivery/TAR_SHA256.txt
    cd $RUN
    find . -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum > SHA256SUMS
"
status_done DONE all
echo "[main] $DOM complete $(date -Is)" | tee -a "$LOGS/main.log"
