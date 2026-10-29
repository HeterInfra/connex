#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
# Entrypoint for one vLLM pipeline stage on Connex (docker/Dockerfile image).
#
# PP_ENGINE selects the vLLM build the image was made with [CONNEX_PP_ENGINE
# baked in at build time, default upstream]:
#   upstream  vLLM main + the three proposed PP changes, plugin
#             connex.adapters.vllm.pp (integrations/vllm/README.md). Stage 0
#             serves the API and drives one replica of every downstream stage;
#             downstream stages run `python -m connex.adapters.vllm.pp.stage`.
#   fork      legacy external-PP fork (--pipeline-stage-mode external) with
#             the Connex front-router; see "Legacy" in docs/deploy.md.
#
# Required env:
#   STAGE          stage index (0 = API/scheduler stage)
#   NUM_STAGES     total number of stages
#   PP_STAGES      (upstream, stage 0) JSON list, one entry per downstream
#                  stage, of its replicas' control addresses HOST:PORT; host
#                  names are resolved to addresses at start. Exported as
#                  CONNEX_PP_STAGES.
#   ROUTER_ADDR    (fork) Connex front-router API address HOST:PORT
# Upstream env (defaults in brackets):
#   CONTROL_PORT   control port of a downstream stage [47000 + 1000*STAGE + 16*REPLICA]
#   DATA_PORT      data port of TP rank 0 of a downstream stage [CONTROL_PORT + 1];
#                  TP rank r listens on DATA_PORT + r (connex.adapters.vllm.pp.ports)
#   NUM_GPU_BLOCKS --num-gpu-blocks-override, the same on every stage [8192]
# Common env (defaults in brackets):
#   BACKEND        (fork) connexte | connex [connexte]
#   REPLICA        replica index of this stage [0]
#   MODEL_PATH     model directory inside the container, or a Hugging Face
#                  model id downloaded at start (HF_TOKEN for gated models) [/model]
#   MODEL_NAME     served model name [connex-model]
#   NUM_LAYERS     (fork) model decoder layers, split evenly across stages [32]
#   LAYER_RANGE    (fork) explicit "lo-hi" layer range (overrides NUM_LAYERS split)
#   API_PORT       stage-0 OpenAI port / stage>0 metrics port [5100 + REPLICA]
#   BASE_PORT      (fork) Connex pipeline router port [25500 + 10*REPLICA]
#   HOST_IP        address peers use to reach this stage [first `hostname -I`]
#   GPU_MEM_UTIL   --gpu-memory-utilization [0.6]
#   TP             tensor-parallel size of this stage [1]; TP rank 0 is the
#                  stage's pipeline endpoint (fork: needs CONNEX_PP_INFLIGHT=1;
#                  TP GPUs in CUDA_VISIBLE_DEVICES, e.g. GPU_SELECT=none)
#   EXTRA_ARGS     extra arguments for vLLM
# GPU selection (GPU_SELECT):
#   none       keep CUDA_VISIBLE_DEVICES as given (docker --gpus / k8s device plugin)
#   auto       pick the GPU with the most free memory >= MIN_FREE_MIB [30000];
#              use with NVIDIA_VISIBLE_DEVICES=all when GPUs are shared
#   nic-local  like auto but prefer GPUs whose PCIe path to NIC0 is not "SYS"
#              (GPUDirect RDMA is slow across sockets); PREFERRED_GPU pins one
#              index and waits up to PREFERRED_GPU_WAIT_S [120] for it to free up
#   [default: none if CUDA_VISIBLE_DEVICES is set, else auto]
set -euo pipefail

: "${STAGE:?STAGE is required}" "${NUM_STAGES:?NUM_STAGES is required}"
PP_ENGINE=${PP_ENGINE:-${CONNEX_PP_ENGINE:-upstream}}
case "$PP_ENGINE" in
  upstream) ;;
  fork) : "${ROUTER_ADDR:?ROUTER_ADDR is required with PP_ENGINE=fork}" ;;
  *) echo "PP_ENGINE must be upstream or fork, got '$PP_ENGINE'" >&2; exit 1 ;;
esac
BACKEND=${BACKEND:-connexte}
REPLICA=${REPLICA:-0}
MODEL_PATH=${MODEL_PATH:-/model}
MODEL_NAME=${MODEL_NAME:-connex-model}
NUM_LAYERS=${NUM_LAYERS:-32}
API_PORT=${API_PORT:-$((5100 + REPLICA))}
BASE_PORT=${BASE_PORT:-$((25500 + 10 * REPLICA))}
HOST_IP=${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
HOST_IP=${HOST_IP:-127.0.0.1}

# vLLM's compiled extensions (the fork's in particular) dlopen libcudart from the pip CUDA runtime
# wheel; it is not on the default search path in every base image.
cudart=$(python3 -c 'import nvidia.cuda_runtime,os;print(os.path.join(list(nvidia.cuda_runtime.__path__)[0],"lib"))' 2>/dev/null || true)
[ -n "$cudart" ] && export LD_LIBRARY_PATH=$cudart${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ---- GPU selection -------------------------------------------------------
GPU_SELECT=${GPU_SELECT:-$([ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo none || echo auto)}
if [ "$GPU_SELECT" != none ]; then
  need_mib=${MIN_FREE_MIB:-30000}
  nic_local=""
  if [ "$GPU_SELECT" = nic-local ]; then
    nic_local=$(nvidia-smi topo -m 2>/dev/null |
      awk 'NR==1{for(i=1;i<=NF;i++) if($i ~ /NIC0/) c=i} /^GPU[0-9]/{if(c && $(c+1)!="SYS") print substr($1,4)}' | tr '\n' ' ')
  fi
  gpu="" free=""
  # Stages of one node usually start together and all see the same free GPUs
  # (vLLM allocates only after loading weights), so "freest GPU" alone makes
  # them pick the same one. Spread them: among the GPUs with enough free
  # memory (NIC-local ones first), ordered by index so every pod sees the same
  # order even after a sibling has started allocating, take number
  # (STAGE + NUM_STAGES*REPLICA) mod #candidates.
  slot=$((STAGE + NUM_STAGES * ${REPLICA:-0}))
  read -r gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' |
    awk -v local="$nic_local" -v need="$need_mib" 'BEGIN{n=split(local,a," "); for(i=1;i<=n;i++) L[a[i]]=1}
         $2>=need {print ($1 in L ? 1 : 0), $2, $1}' | sort -k1,1nr -k3,3n |
    awk -v slot="$slot" '{g[NR]=$1; c[NR]=$3" "$2} END{if(NR){top=0; for(i=1;i<=NR;i++) if(g[i]==g[1]) top++;
         print c[1 + slot % top]}}') || true
  if [ -n "${PREFERRED_GPU:-}" ]; then
    # Wait for the preferred GPU (a restarting predecessor may still hold it)
    # rather than falling back onto a GPU a sibling replica is about to use.
    pfree=""
    for _ in $(seq "${PREFERRED_GPU_WAIT_S:-120}"); do
      pfree=$(nvidia-smi -i "$PREFERRED_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
      [ -n "$pfree" ] && [ "$pfree" -ge "$need_mib" ] && break
      sleep 1
    done
    if [ -n "$pfree" ] && [ "$pfree" -ge "$need_mib" ]; then gpu=$PREFERRED_GPU; free=$pfree; fi
  fi
  [ -n "$gpu" ] || { echo "no GPU with >= $need_mib MiB free"; exit 1; }
  echo "picked GPU $gpu with $free MiB free (mode $GPU_SELECT; NIC-local: ${nic_local:-n/a})"
  export CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

python3 -c "import connex; print('connex', connex.__file__)" || true

# ---- upstream vLLM: plugin connex.adapters.vllm.pp -------------------------
if [ "$PP_ENGINE" = upstream ]; then
  export VLLM_HOST_IP=$HOST_IP CONNEX_RS_LOCAL_IP=${CONNEX_RS_LOCAL_IP:-$HOST_IP}
  export CONNEX_PP_REPLICA=$REPLICA
  args=(--served-model-name "$MODEL_NAME" --pipeline-parallel-size "$NUM_STAGES"
        --pipeline-stage-rank "$STAGE" --tensor-parallel-size "${TP:-1}"
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" --num-gpu-blocks-override "${NUM_GPU_BLOCKS:-8192}"
        --pp-group-cls connex.adapters.vllm.pp.group.ConnexPPGroup
        --worker-cls connex.adapters.vllm.pp.worker.ConnexPPWorker
        --distributed-executor-backend connex.adapters.vllm.pp.executor.ConnexPPExecutor)
  if [ "$STAGE" -eq 0 ]; then
    : "${PP_STAGES:?PP_STAGES is required on stage 0 (JSON list of downstream replicas HOST:PORT)}"
    # connex-rs connects to addresses: resolve host names (e.g. k8s Services).
    CONNEX_PP_STAGES=$(python3 - "$PP_STAGES" <<'PY'
import json, socket, sys, time
def res(hp):
    h, p = hp.rsplit(":", 1)
    for _ in range(120):
        try:
            return f"{socket.gethostbyname(h)}:{p}"
        except OSError:
            time.sleep(1)
    sys.exit(f"cannot resolve {h}")
print(json.dumps([[res(hp) for hp in st] for st in json.loads(sys.argv[1])]))
PY
)
    export CONNEX_PP_STAGES
    echo "stage 0: downstream replicas $CONNEX_PP_STAGES"
    # shellcheck disable=SC2086
    exec vllm serve "$MODEL_PATH" --port "$API_PORT" "${args[@]}" ${EXTRA_ARGS:-}
  fi
  # One port layout everywhere (connex/adapters/vllm/pp/ports.py): no two
  # (stage, replica, TP rank) share a port on a host network.
  CONTROL_PORT=${CONTROL_PORT:-$(python3 -m connex.adapters.vllm.pp.ports control "$STAGE" "$REPLICA")}
  export CONNEX_PP_DATA_PORT=${DATA_PORT:-$((CONTROL_PORT + 1))}
  # shellcheck disable=SC2086
  exec python3 -m connex.adapters.vllm.pp.stage --model "$MODEL_PATH" "${args[@]}" \
    --connex-control-port "$CONTROL_PORT" --connex-replica "$REPLICA" ${EXTRA_ARGS:-}
fi

# ---- legacy external-PP fork: stage identity / data plane -------------------
export PIPELINE_STAGE_IDX=$STAGE PIPELINE_REPLICA_IDX=$REPLICA
export SERVED_MODEL_NAME=$MODEL_NAME VLLM_HOST_IP=$HOST_IP MOONCAKE_LOCAL_IP=${MOONCAKE_LOCAL_IP:-$HOST_IP}
# With several NICs Mooncake advertises the first one for its handshake RPC;
# pin it to the address peers can actually reach.
export MC_TCP_BIND_ADDRESS=${MC_TCP_BIND_ADDRESS:-$HOST_IP}
if [ "$BACKEND" = connexte ]; then
  export MOONCAKE_METADATA_SERVER=${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}
  export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}
fi

if [ -n "${LAYER_RANGE:-}" ]; then
  range=$LAYER_RANGE
else
  per=$((NUM_LAYERS / NUM_STAGES)); lo=$((STAGE * per)); hi=$((lo + per))
  [ "$STAGE" -eq $((NUM_STAGES - 1)) ] && hi=$NUM_LAYERS
  range="$lo-$hi"
fi
if [ "$STAGE" -eq 0 ]; then role=(--port "$API_PORT"); else role=(--external-pp-worker --metrics-port "$API_PORT"); fi
# shellcheck disable=SC2086
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" --pipeline-stage-mode external \
  --pipeline-stage-idx "$STAGE" --pipeline-total-stages "$NUM_STAGES" --pipeline-layer-range "$range" \
  --tensor-parallel-size "${TP:-1}" --pipeline-comm-backend "$BACKEND" --gpu-memory-utilization "${GPU_MEM_UTIL:-0.6}" \
  --pipeline-router-port "$BASE_PORT" --pipeline-front-router-addr "$ROUTER_ADDR" "${role[@]}" ${EXTRA_ARGS:-}
