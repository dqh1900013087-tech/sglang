#!/usr/bin/env bash
# =============================================================================
# 启动脚本：DeepSeek-V4 (dsv4) + Mooncake L3 backend + MemFabric transport（A5）
#
# 前置：
#   - mooncake 已安装（构建时 USE_TENT=ON + USE_MEMFABRIC=ON）
#   - memfabric_hybrid 已安装（需含 batch API：smem_trans_batch_copy 等）
#   - NPU 容器内运行，CANN set_env.sh 可用
#
# 用法：
#   bash /home/dqh/run_mooncake.sh
#   常用可覆盖项：
#     MODEL=/path/to/model   DEVICES=0,1,2,3   PORT=8001   MASTER_PORT=50052
#     TP_SIZE=4   DP_SIZE=4   MEM_FRACTION=0.72   STORE_PROTOCOL=tcp|ascend   GLOBAL_SEG=8gb
#     CHUNKED=128            # = page_size 时 cold 与命中位一致（默认 65536，性能优先）
#     SGLANG_EXTRA_ARGS="--disable-cuda-graph"
#
# 注意：本脚本**不**传 --disable-radix-cache —— HiCache 的 L1/L2/L3 依赖 radix cache。
# =============================================================================
set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${LOG_DIR:-$DIR/logs}"
mkdir -p "$LOG"

MODEL=${MODEL:-/mnt/share/weight/DeepSeek-V4-Flash}
[ -d "$MODEL" ] || { echo "ERROR: 模型目录不存在: ${MODEL}（用 MODEL=/path 指定）" >&2; exit 1; }
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
MASTER_PORT=${MASTER_PORT:-50052}
MASTER_METRICS_PORT=${MASTER_METRICS_PORT:-9004}
DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7}
MF_PROTOCOL=${MF_PROTOCOL:-device_urma}      # sdma / device_rdma / host_rdma ...
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-4}
PAGE_SIZE=${PAGE_SIZE:-128}
MEM_FRACTION=${MEM_FRACTION:-0.72}
MAX_RUNNING=${MAX_RUNNING:-64}
CHUNKED=${CHUNKED:-512}
MAX_PREFILL=${MAX_PREFILL:-65536}
STORE_PROTOCOL=${STORE_PROTOCOL:-tcp}        # tcp=堆内存；ascend=aclrtMallocHost
GLOBAL_SEG=${GLOBAL_SEG:-8gb}
SGL_ROOT=${SGL_ROOT:-/workspace/sglang}
SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-}

HICACHE_SIZE=${HICACHE_SIZE:-}
if [ -n "$HICACHE_SIZE" ]; then
  HICACHE_ARGS=(--hicache-size "$HICACHE_SIZE")
else
  HICACHE_ARGS=(--hicache-ratio "${HICACHE_RATIO:-0.5}")
fi

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
sysctl -w vm.swappiness=0 >/dev/null 2>&1 || true
sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

# ---- TENT 配置（内嵌，缺失时自动生成） ----
TENT_CONF="$DIR/tent_memfabric_dsv4.json"
if [ ! -f "$TENT_CONF" ]; then
  cat > "$TENT_CONF" <<'JSON'
{
    "local_segment_name": "",
    "metadata_type": "p2p",
    "metadata_servers": "127.0.0.1:2379",
    "rpc_server_hostname": "127.0.0.1",
    "rpc_server_port": 0,
    "log_level": "info",
    "transports": {
        "memfabric": {
            "enable": true
        }
    },
    "policy": [
        {
            "name": "memfabric_memory",
            "segment_type": "memory",
            "transports": ["memfabric"]
        }
    ]
}
JSON
  echo "==> 已生成 $TENT_CONF"
fi

# ---- CANN + MemFabric 运行时环境 ----
for c in /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
         /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash \
         /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
         /usr/local/Ascend/nnal/atb/set_env.sh; do
  if [ -f "$c" ]; then
    # shellcheck disable=SC1090
    source "$c" >/dev/null 2>&1 || true
  fi
done

MF_LIB="$(python3 -c 'import memfabric_hybrid, os; print(os.path.join(os.path.dirname(memfabric_hybrid.__file__), "lib"))' 2>/dev/null | tail -1 || true)"
[ -n "$MF_LIB" ] || { echo "ERROR: 未安装 memfabric_hybrid（需含 batch API 的版本）" >&2; exit 1; }
export LD_LIBRARY_PATH="$MF_LIB:$LD_LIBRARY_PATH"
export MC_USE_TENT=1
export MC_TENT_CONF="$TENT_CONF"
export MF_PROTOCOL
export MF_HYBM_URMA_SWAP_SPACE_SIZE=${MF_HYBM_URMA_SWAP_SPACE_SIZE:-512}
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=${MF_HYBM_RDMA_SWAP_SPACE_SIZE:-512}
export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export PYTHONPATH="$SGL_ROOT/python:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export PYTHONUNBUFFERED=1
export ASCEND_USE_FIA=1
export SGLANG_HICACHE_MOONCAKE_REUSE_TE=0

# HCCL
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_INTRA_DMA_ENABLE=1

# dsv4 专用开关
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

# spec / 性能
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1

echo "==> DIR=$DIR"
echo "==> SGL_ROOT=$SGL_ROOT"
echo "==> MF_PROTOCOL=$MF_PROTOCOL  store_protocol=$STORE_PROTOCOL  devices=$DEVICES  tp=$TP_SIZE  dp=$DP_SIZE"
echo "==> chunked=$CHUNKED  page_size=$PAGE_SIZE  mem_fraction=$MEM_FRACTION  max_prefill=$MAX_PREFILL"
echo "==> master=127.0.0.1:$MASTER_PORT  server=:$PORT  model=$MODEL"
[ -n "$SGLANG_EXTRA_ARGS" ] && echo "==> SGLANG_EXTRA_ARGS=$SGLANG_EXTRA_ARGS"

# ---- 1. mooncake master ----
port_in_use() {
  python3 - "$1" <<"PY"
import socket, sys
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    print("up"); sys.exit(0)
except OSError:
    sys.exit(1)
PY
}
if port_in_use "$MASTER_PORT"; then
  echo "==> mooncake master 已在端口 $MASTER_PORT 运行，复用"
else
  nohup mooncake_master --rpc_port "$MASTER_PORT" --metrics_port "$MASTER_METRICS_PORT" \
      --eviction_high_watermark_ratio=0.95 > "$LOG/master.log" 2>&1 &
  echo "==> mooncake master pid=$! (rpc $MASTER_PORT, metrics $MASTER_METRICS_PORT)"
  sleep 5
fi

# ---- 2. SGLang server（Mooncake L3 + MemFabric transport） ----
# shellcheck disable=SC2086
nohup python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --attention-backend dsv4 \
  --device npu \
  --tp-size "$TP_SIZE" \
  --enable-dp-attention \
  --dp-size 4 \
  --enable-dp-lm-head \
  --page-size "$PAGE_SIZE" \
  --trust-remote-code \
  --watchdog-timeout 9000 \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNKED" \
  --max-prefill-tokens "$MAX_PREFILL" \
  --cuda-graph-bs-decode 1 2 4 8 10 16 \
  --kv-cache-dtype auto \
  --reasoning-parser deepseek-v4 \
  --enable-hierarchical-cache \
  --hicache-io-backend kernel_ascend \
  --enable-cache-report \
  --hicache-storage-prefetch-policy "${HICACHE_PREFETCH_POLICY:-wait_complete}" \
  --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_through}" \
  "${HICACHE_ARGS[@]}" $SGLANG_EXTRA_ARGS \
  --hicache-storage-backend mooncake \
  --hicache-storage-backend-extra-config "{\"master_server_address\":\"127.0.0.1:${MASTER_PORT}\",\"metadata_server\":\"P2PHANDSHAKE\",\"protocol\":\"${STORE_PROTOCOL}\",\"global_segment_size\":\"${GLOBAL_SEG}\",\"local_hostname\":\"127.0.0.1\"}" \
  > "$LOG/server.log" 2>&1 &
echo "==> sglang pid=$!  port=$PORT  log=$LOG/server.log"
echo "==> 等待就绪（3-10 分钟）: curl -s http://127.0.0.1:$PORT/ready"
