#!/usr/bin/env bash
# Second one-off packaging pass (2026-11-04): include VERIFY_SELECTION_val.json
# and the report's selection_val field in the delivery, refresh code_snapshot,
# tar, manifest and HANDOFF.  No export/feature/head/eval/freeze is recomputed.
set -euo pipefail

ROOT=/root/autodl-tmp/d4p_trainval_20261104/C
CODE=$ROOT/code
PY=/root/miniconda3/envs/des4pos/bin/python

for DOM in npm3d s3dis; do
    RUN=$ROOT/$DOM
    echo "=== $DOM"
    "$PY" "$CODE/d4p_report_trainval.py" --domain "$DOM" --run_dir "$RUN"
    OLD_TAR_SHA=$(awk '{print $1}' "$RUN/delivery/TAR_SHA256.txt")

    rm -rf "$RUN/code_snapshot"
    mkdir -p "$RUN/code_snapshot/adapt"
    cp "$CODE"/*.py "$CODE"/*.sh "$RUN/code_snapshot/"
    cp "$ROOT"/records/*.json "$RUN/code_snapshot/" 2>/dev/null || true
    cp /root/autodl-tmp/des4pos_corrected_C_20261102/code/adapt/*.py \
        "$RUN/code_snapshot/adapt/" 2>/dev/null || true
    find /root/autodl-tmp/des4pos_corrected_C_20261102/code/core -maxdepth 1 \
        -iname '*manifest*' -exec cp {} "$RUN/code_snapshot/" \; 2>/dev/null || true

    rm -f "$RUN/delivery/${DOM}_corrected_scar_trainval.tar"
    ( cd "$ROOT" && tar -cf "$ROOT/.${DOM}_pkg.tar" --exclude="$DOM/delivery" "$DOM" )
    mv "$ROOT/.${DOM}_pkg.tar" "$RUN/delivery/${DOM}_corrected_scar_trainval.tar"
    sha256sum "$RUN/delivery/${DOM}_corrected_scar_trainval.tar" \
        > "$RUN/delivery/TAR_SHA256.txt"
    NEW_TAR_SHA=$(awk '{print $1}' "$RUN/delivery/TAR_SHA256.txt")

    cat >> "$RUN/delivery/REPACKAGE.md" <<EOF

## Second pass (same date)

Reason: an independent VAL epoch-selection verifier
(verify/VERIFY_SELECTION_val.json) was run after the chain, and REPORT.json now
records its status (verification.selection_val). code_snapshot, tar and
SHA256SUMS were rebuilt to include both. No export, feature, head, evaluation
or freeze artefact was recomputed.

Previous tar SHA-256: $OLD_TAR_SHA
New tar SHA-256:      $NEW_TAR_SHA
REPORT.json (final):  $(sha256sum "$RUN/REPORT.json" | awk '{print $1}')
EOF

    ( cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
        xargs -0 sha256sum > SHA256SUMS )
    echo "  tar $NEW_TAR_SHA"
    echo "  manifest $(sha256sum "$RUN/SHA256SUMS" | awk '{print $1}')"
done

"$PY" - "$ROOT" <<'PYEOF'
import hashlib, json, os, sys
root = sys.argv[1]
p = os.path.join(root, "HANDOFF.json")
h = json.load(open(p))
for d in h["domains"]:
    rec = h["domains"][d]
    rec["tar_sha256"] = open(os.path.join(root, d, "delivery", "TAR_SHA256.txt")).read().split()[0]
    rec["sha_manifest"] = os.path.join(root, d, "SHA256SUMS")
    rec["sha_manifest_sha256"] = hashlib.sha256(
        open(os.path.join(root, d, "SHA256SUMS"), "rb").read()).hexdigest()
    rec["report_sha256"] = hashlib.sha256(
        open(os.path.join(root, d, "REPORT.json"), "rb").read()).hexdigest()
    rec["selection_verify_sha256"] = hashlib.sha256(
        open(os.path.join(root, d, "verify", "VERIFY_SELECTION_val.json"), "rb").read()).hexdigest()
h["note"] = ("rebuilt 2026-11-04 after the TRAIN/VAL chain; includes the independent "
             "epoch-selection verification; no TEST read or armed")
json.dump(h, open(p, "w"), indent=2)
print({d: (h["domains"][d]["tar_sha256"][:12],
           h["domains"][d]["sha_manifest_sha256"][:12],
           h["domains"][d]["selection_verify_sha256"][:12]) for d in h["domains"]})
PYEOF
