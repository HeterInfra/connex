# Contributing to Connex

## Development setup

```bash
pip install -e ".[dev]"              # add ".[ucx,mooncake]" for the data planes
python -m pytest -q -p no:cacheprovider tests
python scripts/check_release.py      # must pass before every commit
```

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
(`connexte`/`connex`), the Mooncake/UCX transport, and the stage logs.
