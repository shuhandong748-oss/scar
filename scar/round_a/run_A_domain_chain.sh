#!/bin/bash
# A-domain FEATURE_COORD_FIX_v1 chain: corrected-backbone TRAIN/VAL export ->
# corrected + old-defect features (diagnostic) -> full comparison -> geometry
# validation -> 3-seed scorer -> VAL eval -> independent verification -> freeze.
# TRAIN/VAL only; never touches TEST; no efficiency work; no shutdown.
set -uo pipefail

ROOT=/root/autodl-tmp/d4p_trainval_20261104/A
CODE=$ROOT/code
PY=/root/miniconda3/envs/des4pos/bin/python
LOCKDIR=/root/autodl-tmp/efficiency_audit_20261029
LOCK=$LOCKDIR/GPU.lock
CORR=/root/autodl-tmp/des4pos_corrected_A_20261102
DOMAIN=${DOMAIN:?DOMAIN must be pandaset|nuscenesfull}

case "$DOMAIN" in
  pandaset)
    RUN=$ROOT/pandaset
    D_RUN=$CORR/pandaset_s0
    CKPT_DIR=$D_RUN/ckpt/PandaSetPose-50-25-25
    CONFIG=$CKPT_DIR/coarse_corrected_s0_20ep/config.json
    KNOWN=$CKPT_DIR/coarse_corrected_s0_20ep/known_classes.json
    PRIOR=/root/autodl-tmp/des4pos_pandaset_v2/runs/pandaset_bb0_seed0/ste_stage1/ste_stage1_prior.pth
    COARSE=$D_RUN/coarse/coarse_best_light.pth
    FINE=$D_RUN/fine/fine_best_light.pth
    ;;
  nuscenesfull)
    RUN=$ROOT/nuscenesfull
    D_RUN=$CORR/nuscenesfull_s0
    CKPT_DIR=$D_RUN/ckpt/nuScenesPose-50-25-25_30-10
    CONFIG=$CKPT_DIR/coarse_corrected_s0_20ep/config.json
    KNOWN=$CKPT_DIR/coarse_corrected_s0_20ep/known_classes.json
    PRIOR=/root/autodl-tmp/des4pos_nuscenesfull_v2/runs/nusc_full_bb0_seed0/ste_stage1/ste_stage1_prior.pth
    COARSE=$D_RUN/coarse/coarse_best_light.pth
    FINE=$D_RUN/fine/fine_best_light.pth
    ;;
  *) echo "unknown DOMAIN=$DOMAIN" >&2; exit 2 ;;
esac

LOG=$ROOT/logs
mkdir -p "$LOG" "$RUN"/{export,features,old_features,heads,eval_val,verify,records}
export PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT:$CODE:$CORR
export CUDA_VISIBLE_DEVICES=0
export T2L_NUSC_ROOT=/root/autodl-tmp/text2loc_nuscenes/data/nuScenesPose-50-25-25_30-10

PROTO_SHA=$(sha256sum "$ROOT/records/PROTOCOL_FREEZE.json" | cut -d' ' -f1)
echo "protocol_sha=$PROTO_SHA domain=$DOMAIN"

step_done() {
  [ -f "$RUN/records/steps.jsonl" ] && \
    grep -q "\"step\":\"$1\",\"exit\":0" "$RUN/records/steps.jsonl"
}

step() {
  local name=$1; shift
  if step_done "$name"; then
    echo "=== STEP $name skip (already exit=0)" >&2
    return 0
  fi
  local t0 rc t1
  t0=$(date +%s)
  echo "=== STEP $name start $(date -Is)" >&2
  "$@" >> "$LOG/$DOMAIN.$name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  printf '{"step":"%s","exit":%d,"start":%d,"end":%d}\n' \
    "$name" "$rc" "$t0" "$t1" >> "$RUN/records/steps.jsonl"
  echo "=== STEP $name exit=$rc seconds=$((t1 - t0))" >&2
  if [ "$rc" -ne 0 ]; then
    echo "CHAIN STOP domain=$DOMAIN step=$name rc=$rc" >&2
    exit "$rc"
  fi
}

# ---- preflight: identity, disk, GPU, lock ---------------------------------
"$PY" - "$DOMAIN" "$RUN" "$LOCK" "$COARSE" "$FINE" <<'PYEOF' || exit 3
import hashlib, json, os, shutil, socket, subprocess, sys, time
domain, run, lock, coarse, fine = sys.argv[1:6]
du = shutil.disk_usage("/root/autodl-tmp")
apps = subprocess.run(["nvidia-smi",
                       "--query-compute-apps=pid,process_name,used_memory",
                       "--format=csv,noheader"], capture_output=True,
                      text=True).stdout.strip()
name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,uuid",
                       "--format=csv,noheader"], capture_output=True,
                      text=True).stdout.strip()
lock_free = subprocess.run(["flock", "-n", lock, "true"]).returncode == 0
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
info = {"domain": domain, "hostname": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": name, "compute_apps": apps,
        "free_gib": round(du.free / 1024 ** 3, 2),
        "gpu_lock": lock, "lock_free_at_start": bool(lock_free),
        "python": sys.version,
        "coarse": {"path": coarse, "sha256": sha(coarse)},
        "fine": {"path": fine, "sha256": sha(fine)}}
start = os.path.join(run, "records", "START_BINDING.json")
with open(os.path.join(run, "records", "chain_starts.jsonl"), "a") as f:
    f.write(json.dumps(info) + "\n")
if not os.path.exists(start):           # preserve the original start binding
    json.dump(info, open(start, "w"), indent=2)
if info["free_gib"] < 6.0:
    raise SystemExit(f"insufficient disk: {info['free_gib']} GiB")
if apps:
    raise SystemExit(f"GPU has compute apps: {apps}")
print(json.dumps(info, indent=2))
PYEOF

# ---- exports (canonical per-GPU flock) ------------------------------------
step export_train env GPU_LOCK_WAIT=1800 "$LOCKDIR/lock_gpu.sh" \
  "$PY" "$CODE/export_candidates_coordfix_v1.py" \
  --domain "$DOMAIN" --config "$CONFIG" --coarse_ckpt "$COARSE" \
  --fine_ckpt "$FINE" --known_classes "$KNOWN" \
  --ste_stage1_path "$PRIOR" --split train --out_dir "$RUN/export/train"

step export_val env GPU_LOCK_WAIT=1800 "$LOCKDIR/lock_gpu.sh" \
  "$PY" "$CODE/export_candidates_coordfix_v1.py" \
  --domain "$DOMAIN" --config "$CONFIG" --coarse_ckpt "$COARSE" \
  --fine_ckpt "$FINE" --known_classes "$KNOWN" \
  --ste_stage1_path "$PRIOR" --split val --out_dir "$RUN/export/val"

step verify_export_train "$PY" "$CODE/verify_export_coordfix_v1.py" \
  --domain "$DOMAIN" --split train --export_dir "$RUN/export/train" \
  --out "$RUN/verify/VERIFY_EXPORT_train.json"
step verify_export_val "$PY" "$CODE/verify_export_coordfix_v1.py" \
  --domain "$DOMAIN" --split val --export_dir "$RUN/export/val" \
  --out "$RUN/verify/VERIFY_EXPORT_val.json"

# ---- FEATURE_COORD_FIX_v1 builds ------------------------------------------
for split in train val; do
  step "features_v1_$split" "$PY" "$CODE/build_ht_t2l_coordfix_v1.py" \
    --domain "$DOMAIN" --split "$split" \
    --export_dir "$RUN/export/$split" --out "$RUN/features/$split.npz"
  step "features_old_$split" "$PY" "$CODE/build_ht_t2l_coordfix_v1.py" \
    --domain "$DOMAIN" --split "$split" \
    --export_dir "$RUN/export/$split" --out "$RUN/old_features/$split.npz" \
    --old-defect
done

for split in train val; do
  step "compare_$split" "$PY" "$CODE/compare_feature_versions.py" \
    --old "$RUN/old_features/$split.npz" --new "$RUN/features/$split.npz" \
    --out "$RUN/verify/COMPARE_FEATURES_$split.json"
done

step verify_coord_fix_synthetic "$PY" "$CODE/test_coord_fix_synthetic.py" \
  --out "$RUN/verify/VERIFY_COORD_FIX_SYNTHETIC.json"
for split in train val; do
  step "verify_geom_$split" "$PY" "$CODE/verify_coord_fix_geom.py" \
    --domain "$DOMAIN" --split "$split" \
    --export_dir "$RUN/export/$split" --features "$RUN/features/$split.npz" \
    --out "$RUN/verify/VERIFY_COORD_FIX_GEOM_$split.json"
done

# ---- scorer + VAL evaluation + independent verification -------------------
step train_heads "$PY" "$CODE/train_head_coordfix_v1.py" \
  --domain "$DOMAIN" --train_npz "$RUN/features/train.npz" \
  --val_npz "$RUN/features/val.npz" --out_dir "$RUN/heads" \
  --protocol_sha "$PROTO_SHA" \
  --coord_fix_sha "$(sha256sum "$CODE/feature_coord_fix_v1.py" | cut -d' ' -f1)" \
  --export_dir "$RUN/export/val"

step eval_val "$PY" "$CODE/eval_val_coordfix_v1.py" \
  --domain "$DOMAIN" --features "$RUN/features/val.npz" \
  --heads_dir "$RUN/heads" --out_dir "$RUN/eval_val" \
  --protocol_sha "$PROTO_SHA"

step verify_val "$PY" "$CODE/verify_val_coordfix_v1.py" \
  --domain "$DOMAIN" --features "$RUN/features/val.npz" \
  --heads_dir "$RUN/heads" --eval_dir "$RUN/eval_val" --out_dir "$RUN/verify"

step freeze "$PY" "$CODE/freeze_coordfix_v1.py" \
  --domain "$DOMAIN" --run_dir "$RUN" --protocol_sha "$PROTO_SHA"

echo "CHAIN_DONE domain=$DOMAIN" >> "$RUN/records/steps.jsonl"
echo "CHAIN DONE domain=$DOMAIN $(date -Is)"
