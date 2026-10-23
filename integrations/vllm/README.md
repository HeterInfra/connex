# vLLM integration

| File | Purpose |
|---|---|
| `connexte_communicator.py` | ConnexTE: Mooncake TE data plane + Connex control plane (recommended). A re-export shim of `connex.adapters.vllm.connexte`, built on the engine-neutral core `connex.pp` (see `docs/design/engine-contract.md`); needs the Connex package importable |
| `connex_communicator.py` | Connex over UCX (RDMA/TCP) |
| `zeromq_communicator.py` | ZeroMQ base communicator the others build on (same as the fork's copy) |
| `pipeline_comm_metrics.py` | Prometheus metrics for pipeline communication (same as the fork's copy) |
| `patches/` | engine patches for the external-PP fork; `apply_all.py` applies them in order |

## Status

This is a reference integration, not an installable plugin. It targets the
**external-PP vLLM fork at commit `380186d57`** (stages run as separate
processes with `--pipeline-stage-mode external`) and imports vLLM internals
(`vllm.distributed.parallel_state`, `vllm.utils.network_utils`, the v1
worker/scheduler), which change between vLLM releases.

## Installing

The easiest way is the image in `docker/Dockerfile`, which does all of this
at build time. By hand:

1. `pip install connex-ai` (or `pip install -e /path/to/connex` from a
   checkout; the communicators import `connex.*`;
   remove any vendored `vllm/connex` package from the fork so the standalone
   package is used).
2. Copy `connex_communicator.py` and `connexte_communicator.py` into
   `vllm/distributed/device_communicators/`.
3. `python patches/apply_all.py <vllm>/v1/worker/gpu_model_runner.py`
   (patches the runner and its siblings; idempotent).
4. Start stages with `--pipeline-comm-backend connexte` (or `connex`); see
   `docker/launch_stage.sh` for the full command line.

Runtime knobs are listed in the configuration tables of the top-level
`README.md`.
