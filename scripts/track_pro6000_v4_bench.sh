#!/usr/bin/env bash
# Track Pro 6000 downloads -> start SGLang V4 -> run serving bench.
set -u
SSH=(ssh -4 -o ConnectTimeout=15 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p 2208 bodesi@39.183.171.3)
LOG=/tmp/v4_track.log
STATUS=/tmp/v4_track.status
echo RUNNING >"$STATUS"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

remote() { "${SSH[@]}" "$@"; }

ensure_jobs() {
  remote 'bash -s' <<'EOS'
set -u
export PATH=$HOME/.local/bin:$PATH
mkdir -p "$HOME/models" "$HOME/bin" "$HOME/bench"

shards=$(ls "$HOME/models/DeepSeek-V4-Flash-0731"/*.safetensors 2>/dev/null | wc -l)
incomplete=$(ls "$HOME/models/DeepSeek-V4-Flash-0731"/*.incomplete 2>/dev/null | wc -l)
if ! pgrep -f download_v4_ms.py >/dev/null; then
  if [ "$shards" -lt 48 ] || [ "$incomplete" -gt 0 ]; then
    nohup python3 "$HOME/bin/download_v4_ms.py" >>"$HOME/models/download_v4_ms.log" 2>&1 &
    echo restarted_ms=$!
  fi
fi

img=$(sudo docker images 2>/dev/null | grep -c deepseek-v4-blackwell || true)
pulling=0
ps -eo args | grep -q '[d]ocker pull' && pulling=1
pgrep -f 'pull_sglang.sh' >/dev/null && pulling=1
if [ "${img:-0}" -eq 0 ] && [ "$pulling" -eq 0 ]; then
  nohup bash "$HOME/bin/pull_sglang.sh" >>"$HOME/models/pull_wrapper.log" 2>&1 &
  echo restarted_pull=$!
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  sudo modprobe nvidia 2>/dev/null || true
  sudo modprobe nvidia_uvm 2>/dev/null || true
fi

size=$(du -sh "$HOME/models/DeepSeek-V4-Flash-0731" 2>/dev/null | awk '{print $1}')
shards=$(ls "$HOME/models/DeepSeek-V4-Flash-0731"/*.safetensors 2>/dev/null | wc -l)
incomplete=$(ls "$HOME/models/DeepSeek-V4-Flash-0731"/*.incomplete 2>/dev/null | wc -l)
ms=$(pgrep -f download_v4_ms.py >/dev/null && echo 1 || echo 0)
pull=$(ps -eo args | grep -q '[d]ocker pull' && echo 1 || echo 0)
img=$(sudo docker images 2>/dev/null | grep -c deepseek-v4-blackwell || true)
gpu=$(nvidia-smi -L 2>/dev/null | wc -l)
ingest=$(sudo du -sh /var/lib/containerd/io.containerd.content.v1.content/ingest 2>/dev/null | awk '{print $1}')
echo "SIZE=$size SHARDS=$shards INC=$incomplete MS=$ms PULL=$pull IMG=$img GPU=$gpu INGEST=$ingest"
EOS
}

start_server() {
  remote 'bash /home/bodesi/bin/start_sglang_v4.sh'
}

wait_ready() {
  for i in $(seq 1 180); do
    if remote 'curl -sf http://127.0.0.1:30000/v1/models' >/tmp/v4_models.json 2>/tmp/v4_health.err; then
      log "server ready"
      return 0
    fi
    if (( i % 6 == 0 )); then
      log "still waiting ($i); container logs:"
      remote 'sudo docker logs --tail 30 sglang-v4-flash 2>&1 | tail -30' | tee -a "$LOG" || true
    fi
    sleep 20
  done
  return 1
}

run_bench() {
  remote 'bash /home/bodesi/bin/run_bench_v4.sh'
}

log "tracker start"
while true; do
  if ! out=$(ensure_jobs 2>/tmp/v4_track.err); then
    log "ssh fail: $(tail -1 /tmp/v4_track.err)"
    sleep 60
    continue
  fi
  # keep only the summary line
  summary=$(echo "$out" | grep '^SIZE=' | tail -1)
  log "${summary:-$out}"
  SIZE= SHARDS=0 INC=1 MS=0 PULL=0 IMG=0 GPU=0
  eval "$summary" || true

  if [[ "${SHARDS:-0}" -ge 48 && "${INC:-1}" -eq 0 && "${IMG:-0}" -ge 1 && "${GPU:-0}" -ge 1 ]]; then
    log "READY - starting server"
    start_server | tee -a "$LOG" || { echo FAIL_START >"$STATUS"; exit 1; }
    if ! wait_ready; then
      log "server failed to become ready"
      remote 'sudo docker logs --tail 120 sglang-v4-flash 2>&1' | tee -a "$LOG" || true
      echo FAIL_SERVER >"$STATUS"
      exit 1
    fi
    log "running benchmarks"
    if ! run_bench | tee -a "$LOG"; then
      echo FAIL_BENCH >"$STATUS"
      exit 1
    fi
    echo SUCCESS >"$STATUS"
    log "ALL DONE"
    exit 0
  fi
  sleep 90
done
