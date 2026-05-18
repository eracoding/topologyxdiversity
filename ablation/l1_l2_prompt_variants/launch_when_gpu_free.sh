#!/usr/bin/env bash
# Polls the GPU every 2 minutes and launches the A12 sweep the moment
# free memory exceeds the threshold. Designed to fire-and-forget while
# another user's job finishes.
#
# Usage:
#   nohup bash ablation/l1_l2_prompt_variants/launch_when_gpu_free.sh \
#       > ablation/l1_l2_prompt_variants/launch_when_gpu_free.log 2>&1 &
#
# Stop with:
#   kill <pid>   # the pid printed by nohup
#
# Tunables:
THRESHOLD_MIB=40000     # need at least this much free memory before launching
POLL_SECONDS=120        # poll cadence

cd "$(dirname "$0")/../.."   # project root

echo "[$(date)] A12 launcher started — polling GPU every ${POLL_SECONDS}s"
echo "[$(date)] Will launch when free memory >= ${THRESHOLD_MIB} MiB"

while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$FREE" =~ ^[0-9]+$ ]] && [[ "$FREE" -ge "$THRESHOLD_MIB" ]]; then
        echo "[$(date)] GPU free: ${FREE} MiB >= ${THRESHOLD_MIB} MiB — launching A12"
        exec bash ablation/l1_l2_prompt_variants/run_all_variants.sh \
            > ablation/l1_l2_prompt_variants/run_all_variants.log 2>&1
    fi
    echo "[$(date)] GPU free: ${FREE} MiB — waiting"
    sleep "$POLL_SECONDS"
done
