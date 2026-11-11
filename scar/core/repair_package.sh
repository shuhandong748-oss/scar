#!/usr/bin/env bash
# One-off repair (2026-11-04): rebuild code_snapshot + delivery tar + manifest
# and the HANDOFF record for both C domains, after
#   (1) d4p_report_trainval.py was fixed (SCAR_minus_FSR key) and both REPORTS
#       regenerated, and
#   (2) the HANDOFF writer / package self-reference bugs were fixed.
# It does NOT re-export, re-train, re-evaluate or re-freeze anything.
set -euo pipefail

ROOT=/root/autodl-tmp/d4p_trainval_20261104/C
CODE=$ROOT/code
PY=/root/miniconda3/envs/des4pos/bin/python

for DOM in npm3d s3dis; do
    RUN=$ROOT/$DOM
    echo "=== $DOM"
    OLD_TAR_SHA=$(awk '{print $1}' "$RUN/delivery/TAR_SHA256.txt")
    OLD_REPORT_SHA=$(sha256sum "$RUN/REPORT.json" | awk '{print $1}')
    OLD_MANIFEST_SHA=$(sha256sum "$RUN/SHA256SUMS" | awk '{print $1}')

    # code snapshot: same command as the runner's stage F
    rm -rf "$RUN/code_snapshot"
    mkdir -p "$RUN/code_snapshot/adapt"
    cp "$CODE"/*.py "$CODE"/*.sh "$RUN/code_snapshot/"
    cp "$ROOT"/records/*.json "$RUN/code_snapshot/" 2>/dev/null || true
    cp /root/autodl-tmp/des4pos_corrected_C_20261102/code/adapt/*.py \
        "$RUN/code_snapshot/adapt/" 2>/dev/null || true
    find /root/autodl-tmp/des4pos_corrected_C_20261102/code/core -maxdepth 1 \
        -iname '*manifest*' -exec cp {} "$RUN/code_snapshot/" \; 2>/dev/null || true

    # tar: build outside the archived tree, delivery/ excluded
    rm -f "$RUN/delivery/${DOM}_corrected_scar_trainval.tar"
    ( cd "$ROOT" && tar -cf "$ROOT/.${DOM}_pkg.tar" --exclude="$DOM/delivery" "$DOM" )
    mv "$ROOT/.${DOM}_pkg.tar" "$RUN/delivery/${DOM}_corrected_scar_trainval.tar"
    sha256sum "$RUN/delivery/${DOM}_corrected_scar_trainval.tar" \
        > "$RUN/delivery/TAR_SHA256.txt"
    NEW_TAR_SHA=$(awk '{print $1}' "$RUN/delivery/TAR_SHA256.txt")

    cat > "$RUN/delivery/REPACKAGE.md" <<EOF
# Repackaging note ($DOM)

Date: 2026-11-04 (after the TRAIN/VAL chain completed)
No export, feature, head, evaluation or freeze artefact was recomputed.

Reasons:
1. REPORT.json was regenerated because the report script read the comparison key
   SCAR_minus_fsr instead of the actual SCAR_minus_FSR, so the FSR comparison
   block was empty. The report reads frozen eval_val/metrics.json only.
2. The first tar was written inside the tree it archived (delivery/), so it
   carried a stale 7190-byte REPORT.json and a self-reference risk. The tar is
   now built outside the tree and delivery/ is excluded.
3. HANDOFF.json was rebuilt: the original writer received the JSON path as its
   root argument, so all per-domain paths and tar/manifest fields were null.

Old tar SHA-256:      $OLD_TAR_SHA
New tar SHA-256:      $NEW_TAR_SHA
Old REPORT.json SHA:  $OLD_REPORT_SHA
Old SHA256SUMS SHA:   $OLD_MANIFEST_SHA
Repair script:        repair_package.sh (SHA-256 recorded in the local delivery note)
EOF

    ( cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum > SHA256SUMS )
    echo "  old tar $OLD_TAR_SHA"
    echo "  new tar $NEW_TAR_SHA"
    echo "  manifest $(sha256sum "$RUN/SHA256SUMS" | awk '{print $1}')"
done

# HANDOFF: same writer as run_both.sh, with the corrected argument order
"$PY" - "$ROOT" "$ROOT/HANDOFF.json" true <<'PYEOF'
import hashlib, json, os, sys, time
root, out_path, lock_free = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
out = {"status": "COMPLETE_TRAINVAL",
       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "domains": {}, "test_read": False, "test_armed": False,
       "efficiency_started": False, "auto_shutdown": False,
       "gpu_lock_free": lock_free,
       "note": "rebuilt 2026-11-04 after the TRAIN/VAL chain; paths/SHAs are "
               "read from the on-disk artefacts",
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
    rec["sha_manifest_sha256"] = (hashlib.sha256(open(os.path.join(d, "SHA256SUMS"), "rb").read()).hexdigest()
                                  if os.path.isfile(os.path.join(d, "SHA256SUMS")) else None)
    rec["freeze_sha256"] = (json.load(open(os.path.join(d, "FREEZE_LIST.json")))["freeze_sha256"]
                            if os.path.isfile(os.path.join(d, "FREEZE_LIST.json")) else None)
    out["domains"][dom] = rec
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out, indent=2)[:2000])
PYEOF
