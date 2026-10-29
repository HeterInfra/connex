# Security policy

## Supported versions

Security fixes are made on the `main` branch and in the latest release
(currently 0.1.x).

## Reporting a vulnerability

Please do not report security problems in public issues, pull requests or
discussions. Report them privately through GitHub instead: open a private
security advisory at
https://github.com/HeterInfra/connex/security/advisories/new
(the **Security** tab of this repository, then **Report a vulnerability**).

Include what is affected (component, version or commit), how to reproduce it,
and the impact you expect. We will acknowledge the report, keep you informed
while we investigate, and credit you in the advisory unless you prefer
otherwise.

## Scope

connex is research infrastructure meant to run inside a trusted cluster network.
Its control and data planes (ZeroMQ, UCX, HTTP) have no authentication or
encryption of their own; exposing their ports to untrusted networks is outside
the supported deployment model. Reports that show a way to break that
boundary, or bugs that matter even inside it (memory safety, crashes from
malformed messages), are welcome.

## Network exposure

What each component listens on by default (all of it unauthenticated unless
stated):

| Component | Default listen address | Notes |
|---|---|---|
| `connex-directory` (`connex.directory_server`) | `127.0.0.1:5555`, events on port + 1 | pass `--bind <ip>:<port>` to serve other hosts |
| `connex-front-router` | `tcp://*:15555` (API), `tcp://*:15560` (events) | all interfaces; anyone who reaches it can register, deregister or report routes |
| `connex-pd-router` (port 8000), `python -m connex.serving.proxy` (5199) | HTTP on `0.0.0.0` (`--host`) | OpenAI-compatible API; `POST /pd/drain` and `/pd/undrain` are accepted only from loopback or with `Authorization: Bearer <token>` (`--admin-token` / `CONNEX_PD_ADMIN_TOKEN`) |
| vLLM PD connector KV control channel (`ctrl_port + rank`) | the advertised IPv4 `host` (`0.0.0.0` when `host` is a name or IPv6) | override with `kv_connector_extra_config` key `bind_ip` |
| agent session server (`python -m connex.agents.session`) | HTTP on `0.0.0.0:5299` | `POST /sessions/{sid}/migrate` and `/sessions/drain` have no authentication |
| RL weight registry (`connex.rl.registry.RegistryServer`) | HTTP on `0.0.0.0`, any port | JSON only, but any peer can publish a weight version that trainers and inference workers will pull |
| PP stage sockets (`connex.pp`, `connex.adapters.vllm.pp`, the vLLM communicators in `integrations/vllm`) | the stage's host IP (`CONNEX_HOST_IP`, `0.0.0.0` if it cannot be found) | some message headers are decoded with `pickle`, so anyone who can reach a stage port can execute code in that process (the EP mover's IPC descriptors use a restricted unpickler; the UCX transport does not unpickle) |
| connex-rs / UCX / Mooncake data planes | the configured local IP and ports | one-sided reads of registered GPU/host memory |

Keep all of these ports on a private cluster network, and firewall them from
anything you do not trust.
