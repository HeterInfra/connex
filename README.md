<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/connex-logo-dark.svg">
    <img alt="Connex" src="docs/assets/connex-logo.svg" width="320">
  </picture>
</p>

<p align="center"><b>Endpoint mobility primitives for dynamic LLM serving</b><br>
communication endpoints that join, leave, fail and migrate with traffic in flight</p>

<p align="center">
  <a href="https://github.com/HeterInfra/connex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/HeterInfra/connex/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Python 3.10 | 3.11 | 3.12" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab?logo=python&logoColor=white">
  <a href="https://pypi.org/project/connex-ai/"><img alt="PyPI: connex-ai" src="https://img.shields.io/badge/pypi-connex--ai-3775a9?logo=pypi&logoColor=white"></a>
  <a href="https://github.com/HeterInfra/connex-rs"><img alt="data plane: connex-rs (Rust over UCX)" src="https://img.shields.io/badge/data%20plane-connex--rs%20%28Rust%2FUCX%29-0d9488?logo=rust"></a>
  <a href="#citation"><img alt="SIGCOMM 2026" src="https://img.shields.io/badge/paper-SIGCOMM%202026-8a2be2"></a>
</p>

Connex is a communication library for distributed LLM serving in which the
set of peers keeps changing: replicas scale up and down, processes die and
come back, work moves between instances. Every endpoint has a stable logical
identity that a control plane resolves to its current address, and senders
keep in-flight work correct when the endpoint behind an identity changes.
Requests hand over to another replica on failure, and new replicas take
traffic as soon as they join. Connex is the system described in our
SIGCOMM 2026 paper ([Citation](#citation)).

## Supported scenarios

| Scenario | Engines | Status |
|---|---|---|
| 🔗 Multi-replica pipeline parallelism, in-flight failover | vLLM (fork) | ✅ validated |
| 📈 Elastic scale up / down | vLLM | ✅ validated |
| 🔀 [Prefill-decode disaggregation](docs/pd-disaggregation.md), elastic P/D pools | vLLM 0.32, SGLang 0.5.22 | ✅ validated, one host |
| 🧩 [Tensor parallelism](docs/tp.md): a TP group is one endpoint | vLLM, SGLang | ✅ validated, one host |
| 🧠 [Expert parallelism](docs/ep.md) (MoE): expert placement, migration, failover | vLLM 0.32 | ✅ validated, one host |
| 🎯 [RL post-training](docs/rl.md): elastic weight sync | vLLM, SGLang | ✅ validated |
| 🤖 [Agent session migration](docs/agent-migration.md) | vLLM, SGLang | ✅ validated (re-prefill) |
| 🧬 Heterogeneous transports: RDMA, TCP, GPUDirect | any | ✅ validated |
| 🪓 Attention-FFN disaggregation | | 🗺️ roadmap |

PD, TP, EP, RL and agent support run on upstream engines as plugins, without
a fork. Limits per scenario: [Scenarios and status](docs/overview.md#scenarios-and-status).

## Quick start

```bash
pip install connex-ai                 # core (import name: connex)
pip install "connex-ai[rs]"           # + connex-rs, the RDMA data plane
pip install "connex-ai[vllm,rs]"      # + vLLM plugins (PD, TP, EP, RL)
pip install "connex-ai[sglang,rs]"    # + SGLang plugins
```

A CPU-only example: a sender talks to two replicas of one logical group and
keeps working while one dies, rejoins, migrates to a new address and leaves
([`examples/mobile_endpoints.py`](examples/mobile_endpoints.py)):

```python
peers = ElasticPeerSet(owner, transport, stage_idx=1)   # resolved through the directory
peers.send(peers.pick(), b"step")
peers.mark_failed(("demo", 1, 0, 1))    # replica 1 died
peers.send(peers.pick(), b"step-42")    # in-flight step handed over to replica 0
b.register()                            # replica 1 rejoins
a.register(addresses=[new_addr_a])      # replica 0 migrates, same identity
```

Prefill-decode disaggregation on upstream vLLM:

```bash
# prefill and decode instances; the connector is registered by the connex-ai plugin
vllm serve <model> --port 8100 --kv-transfer-config '{"kv_connector":"ConnexConnector",
  "kv_role":"kv_producer","kv_connector_extra_config":{"ctrl_port":17100}}'
vllm serve <model> --port 8200 --kv-transfer-config '{"kv_connector":"ConnexConnector",
  "kv_role":"kv_consumer"}'
connex-pd-router --port 8000 --prefill http://<p-host>:8100 --decode http://<d-host>:8200
```

Every scenario has its own guide (links in the table above). Deployment from
a CPU demo to Kubernetes: [`docs/deploy.md`](docs/deploy.md).

## How it works

<p align="center">
  <img alt="Connex control plane (front router / directory, mobility, step protocol) and the pluggable data plane (connex-rs, Mooncake TE, UCX, ZeroMQ)" src="docs/assets/architecture.svg" width="860">
</p>

Connex separates *who talks to whom, and what happens when that changes*
from *how the bytes move*. The control plane holds logical identities,
leases and membership events. The delivery contract (correlation ids,
idempotent retries, stale-reply filtering) makes it safe to resend in-flight
work to another replica. Bytes move through a pluggable data plane: on RDMA
hosts the default is [**connex-rs**](https://github.com/HeterInfra/connex-rs),
our Rust data plane over UCX, which reports a dead peer within milliseconds
and lets a restarted process rejoin under the same identity. Mooncake
Transfer Engine, UCX and ZeroMQ also work. Details: [`docs/overview.md`](docs/overview.md).

## Results

<p align="center">
  <img alt="Handover time (kill a stage to next token) and steady-state mean latency, connex-rs vs Mooncake TE" src="docs/assets/results-connexrs-vs-mooncake.svg" width="860">
</p>

Llama-3-8B on A40 GPUs (OLMoE-1B-7B for EP); every request completed and no
token differed from the greedy reference in any failure run.

| Scenario | Steady state | Kill a process under load |
|---|---|---|
| Pipeline, 4 stages × 2 replicas | same latency as Mooncake TE | next token 0.01-0.83 s after the kill (Mooncake: 2-3 s) |
| PD disaggregation | on par with NIXL (vLLM) and Mooncake (SGLang) | decode kill: KV re-pulled, max stall 0.63 s |
| Tensor parallelism 2 | PD 1061 tok/s, TPOT 16 ms | one rank killed: whole group out in 0.16-0.57 s, stall 0.4 s |
| Expert parallelism | 2,307 tok/s vs 2,338 for vLLM's EP | rank failure detected in 0.05 s, max stall 0.18 s |
| Churn, replica killed every 60 s | | 99.2-99.6 % goodput, 100 % with the resume proxy |

Full tables, setup and reproduction commands: [`docs/results.md`](docs/results.md).

## Documentation

| | |
|---|---|
| [Overview](docs/overview.md) | abstractions, scenario status, architecture, limitations |
| [Deploy](docs/deploy.md) | CPU demo, docker compose, multi-host, Kubernetes |
| [Configuration](docs/configuration.md) | every environment variable and flag |
| [Results](docs/results.md) | all measurements and how to reproduce them |
| [vLLM integration](integrations/vllm/README.md) | fork for pipeline parallelism, engine patches |
| [Design notes](docs/design/) | rationale and engine contract |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

<!-- release-scan: allow-begin -->
## Citation

If you use Connex in your research, please cite our paper:

```bibtex
@inproceedings{lin2026connex,
  title={Connex: Endpoint Mobility Primitives for Dynamic LLM Serving},
  author={Lin, Yanying and Liu, Vincent and Luo, Tao and Xu, Chengzhong and Ye, Kejiang},
  booktitle={Proceedings of the ACM SIGCOMM 2026 Conference},
  pages={46--59},
  year={2026}
}
```

<!-- release-scan: allow-end -->

## License

Apache-2.0, see [LICENSE](LICENSE).
