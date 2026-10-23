# Contributing to Connex

## Development setup

```bash
git clone --recurse-submodules https://github.com/HeterInfra/connex.git
pip install -e ".[dev]"              # add ".[ucx,mooncake]" for the data planes
python -m pytest -q -p no:cacheprovider tests
python scripts/check_release.py      # must pass before every commit
```

Without connex-rs installed, `CONNEX_PP_BACKEND=auto` falls back to Mooncake
or ZeroMQ and `tests/test_connexrs_backend.py` skips.

## connex-rs (submodule)

The Rust data plane lives in its own repository,
[HeterInfra/connex-rs](https://github.com/HeterInfra/connex-rs), and is
pinned here as the git submodule `third_party/connex-rs`. Changes to
connex-rs go to that repository first; then bump the pin here:

```bash
git submodule update --init third_party/connex-rs
cd third_party/connex-rs && git fetch && git checkout <tag or commit> && cd -
git add third_party/connex-rs && git commit -m "Bump connex-rs to <version>: <why>"
```

Build and test it against the Connex backend tests (needs Rust, maturin and
a UCX; `UCX_HOME` is its prefix, default `/opt/hpcx/ucx`, else `/usr`):

```bash
(cd third_party/connex-rs && maturin build --release -o dist)
pip install third_party/connex-rs/dist/connex_rs-*.whl
UCX_TLS=tcp python -m pytest -q tests/test_connexrs_backend.py
```

To work against a local connex-rs checkout (for example before a change is
pushed), point the submodule at it in your own git config only; never commit
a local path to `.gitmodules`:

```bash
git config submodule.third_party/connex-rs.url /path/to/connex-rs
git -c protocol.file.allow=always submodule update --init third_party/connex-rs
git -C third_party/connex-rs remote set-url origin /path/to/connex-rs
```

The image build uses the submodule when it is checked out, otherwise
`--build-arg CONNEX_RS_REPO=<git URL> --build-arg CONNEX_RS_REF=<tag>`.

Tests that patch the vLLM fork need its source tree:

```bash
git clone <vllm-fork-repo> /path/to/vllm-fork && git -C /path/to/vllm-fork checkout <VLLM_FORK_REF>
VLLM_FORK_DIR=/path/to/vllm-fork python -m pytest -q tests
```

Without `VLLM_FORK_DIR`, ucx-py or GPUs, the affected tests skip.

## Guidelines

- Keep code and docs free of environment-specific data: no host names, IP
  addresses (use `127.0.0.1` or the documentation ranges `192.0.2.0/24`,
  `198.51.100.0/24`, `203.0.113.0/24`), personal paths, registries,
  namespaces or credentials. Make them flags or environment variables with
  neutral defaults. `scripts/check_release.py` enforces this in CI; put any
  extra, private patterns for your own deployment in `.release-patterns.local`
  (ignored by git).
- New runtime knobs must be documented in the configuration tables of
  `README.md`, with their default.
- Engine patches in `integrations/vllm/patches/` must be idempotent, fail
  loudly when their anchor is missing, and come with a test that applies them
  to a copy of the fork source.
- Design notes and measurements go in `docs/design/`: explain why, and how
  to reproduce.
- Commit messages say what changed and why.

## Reporting issues

Include the Connex commit, the fork commit (`VLLM_FORK_REF`), the backend
(`connexte`/`connex`), the data plane (`CONNEX_PP_BACKEND`, and the
connex-rs version if used), the Mooncake/UCX transport, and the stage logs.
