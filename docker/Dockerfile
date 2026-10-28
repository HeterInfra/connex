# Connex + vLLM + connex-rs (default data plane, Rust over UCX; from the
# third_party/connex-rs submodule or CONNEX_RS_REPO) + ucx-py + Mooncake
# Transfer Engine (optional data plane).
#
#   make image IMAGE=<registry>/connex-vllm:<tag>
# or
#   docker build -f docker/Dockerfile -t connex-vllm:dev .
#
# PP_ENGINE selects the vLLM build:
#   upstream (default)  latest vLLM main (VLLM_REPO at VLLM_REF) with the three
#                       pipeline-parallelism changes proposed in RFC
#                       vllm#63818 merged in from VLLM_PP_REPO (branches
#                       VLLM_PP_BRANCHES: vllm#63897, #63898, #63899). Pipeline
#                       stages use the plugin connex.adapters.vllm.pp; no
#                       engine patches. VLLM_INSTALL says how it is installed:
#     precompiled (default)  Python sources of the merged tree, compiled
#                            kernels of the nearest vLLM main nightly wheel
#                            (VLLM_USE_PRECOMPILED=1; the changes are Python only)
#     source                 compile vLLM (slow; CUDA devel base image, MAX_JOBS)
#     none                   keep the vLLM of BASE_IMAGE (an image that already
#                            has the changes)
#   fork (legacy)       the external-PP vLLM fork (VLLM_FORK_REPO, required, at
#                       VLLM_FORK_REF) with Connex's engine patches and
#                       communicators; FORK_INSTALL=overlay copies its Python
#                       tree over BASE_IMAGE's compiled vLLM (the base must match
#                       the fork's torch, e.g. the release the fork is based
#                       on), FORK_INSTALL=source compiles it.
#
# Everything comes from public sources: the base image (BASE_IMAGE, default the
# public vLLM image on Docker Hub, tag latest), vLLM and the PP branches on
# GitHub, PyPI (ucx-py, Mooncake) and this repository (the build context). No
# model is baked in: stages load MODEL_PATH at runtime, either a mounted
# directory or a Hugging Face model id (set HF_TOKEN for gated models and mount
# a cache).

ARG BASE_IMAGE=vllm/vllm-openai:latest

# ---- connex-rs builder: UCX (verbs, rdmacm, CUDA) + Rust + maturin ---------
# connex-rs lives in its own repository (https://github.com/HeterInfra/connex-rs).
# Source, in this order:
#   1. the git submodule third_party/connex-rs in the build context
#      (git submodule update --init), pinned by this repository;
#   2. otherwise CONNEX_RS_REPO at CONNEX_RS_REF (git clone).
# The build fails with a message if neither is available. The UCX in the pip
# wheels (libucx-cu12) has no InfiniBand transports, so connex-rs links a UCX
# built here; it is copied to /opt/ucx in the final image (the wheel finds it
# by rpath). BUILD_CONNEX_RS=0 skips all of this (CONNEX_PP_BACKEND=auto then
# picks Mooncake).
FROM ${BASE_IMAGE} AS connex-rs-build
ARG BUILD_CONNEX_RS=1
ARG CONNEX_RS_REPO=https://github.com/HeterInfra/connex-rs.git
ARG CONNEX_RS_REF=v0.1.0
ARG UCX_VERSION=1.18.1
ARG RUST_TOOLCHAIN=1.86.0
ARG MAX_JOBS=8
ARG PIP_INDEX_URL=
ENV DEBIAN_FRONTEND=noninteractive
COPY third_party/ /src/third_party/
RUN set -eux; mkdir -p /wheels; \
    if [ "${BUILD_CONNEX_RS}" != 1 ]; then exit 0; fi; \
    if [ -f /src/third_party/connex-rs/Cargo.toml ]; then \
        echo "connex-rs: using the third_party/connex-rs submodule"; \
        cp -a /src/third_party/connex-rs /src/connex-rs; \
    else \
        command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git ca-certificates); \
        if [ -z "${CONNEX_RS_REPO}" ] || ! git clone --filter=blob:none "${CONNEX_RS_REPO}" /src/connex-rs \
            || ! git -C /src/connex-rs checkout "${CONNEX_RS_REF}"; then \
            echo "ERROR: connex-rs source unavailable. Run 'git submodule update --init third_party/connex-rs'" \
                 "before building, or pass --build-arg CONNEX_RS_REPO=<git URL> --build-arg CONNEX_RS_REF=<tag or commit>" \
                 "(or --build-arg BUILD_CONNEX_RS=0 to build without connex-rs)." >&2; \
            exit 1; \
        fi; \
    fi; \
    rm -rf /src/third_party /src/connex-rs/target /src/connex-rs/dist
RUN set -eux; mkdir -p /opt/ucx; \
    if [ "${BUILD_CONNEX_RS}" != 1 ]; then exit 0; fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential curl ca-certificates pkg-config \
        libibverbs-dev librdmacm-dev libnuma-dev; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" \
        | tar -xz -C /tmp; \
    cd "/tmp/ucx-${UCX_VERSION}"; \
    cuda=""; [ -f /usr/local/cuda/include/cuda.h ] && cuda="--with-cuda=/usr/local/cuda"; \
    ./contrib/configure-release --prefix=/opt/ucx --enable-mt --without-java --without-go \
        --with-verbs --with-rdmacm $cuda; \
    make -j"${MAX_JOBS}"; make install; cd /; rm -rf "/tmp/ucx-${UCX_VERSION}"; \
    /opt/ucx/bin/ucx_info -v
RUN set -eux; if [ "${BUILD_CONNEX_RS}" != 1 ]; then exit 0; fi; \
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"; \
    pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} "maturin>=1.5,<2"; \
    cd /src/connex-rs; \
    UCX_HOME=/opt/ucx PATH="/root/.cargo/bin:$PATH" maturin build --release -o /wheels

FROM ${BASE_IMAGE}

ARG PP_ENGINE=upstream
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main
# release-scan: allow-begin (public fork that carries the proposed PP branches)
ARG VLLM_PP_REPO=https://github.com/yanyinglin/vllm.git
# release-scan: allow-end
# feat/external-pp-stages is stacked on feat/pluggable-pp-group.
ARG VLLM_PP_BRANCHES="feat/pluggable-pp-group feat/external-pp-stages feat/recompute-lost-kv"
ARG VLLM_INSTALL=precompiled
ARG VLLM_FORK_REPO=
ARG VLLM_FORK_REF=380186d57
ARG FORK_INSTALL=overlay
ARG UCX_PY_PKG=ucx-py-cu12
ARG MOONCAKE_PKG=mooncake-transfer-engine
ARG APPLY_PATCHES=all
ARG MAX_JOBS=8
ARG PIP_INDEX_URL=
# A build-time HTTP proxy can be passed as a standard build arg; it is not baked in.

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN command -v git >/dev/null || \
    (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)

# ---- vLLM -------------------------------------------------------------------
RUN set -eux; \
    case "${PP_ENGINE}" in \
    upstream) \
        if [ "${VLLM_INSTALL}" = none ]; then exit 0; fi; \
        git clone --filter=blob:none "${VLLM_REPO}" /opt/vllm; cd /opt/vllm; \
        git checkout "${VLLM_REF}"; base=$(git rev-parse HEAD); \
        git -c user.name=build -c user.email=build@localhost fetch --filter=blob:none "${VLLM_PP_REPO}" \
            $(for b in ${VLLM_PP_BRANCHES}; do printf '%s:refs/pp/%s ' "$b" "$b"; done); \
        for b in ${VLLM_PP_BRANCHES}; do \
            git -c user.name=build -c user.email=build@localhost merge --no-edit "refs/pp/$b" || \
              { echo "ERROR: ${VLLM_PP_REPO} $b does not merge into ${VLLM_REF} ($base); pass --build-arg VLLM_REF=<an older main commit>" >&2; exit 1; }; \
        done; \
        git log --oneline -n 8; \
        if [ "${VLLM_INSTALL}" = source ]; then \
            MAX_JOBS=${MAX_JOBS} pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} --no-build-isolation -e . ; \
        else \
            VLLM_USE_PRECOMPILED=1 \
                pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} -e . ; \
        fi ;; \
    fork) \
        if [ -z "${VLLM_FORK_REPO}" ]; then \
            echo "ERROR: PP_ENGINE=fork needs --build-arg VLLM_FORK_REPO=<git URL of the external-PP vLLM fork> (see docs/deploy.md, 'Legacy: external-PP fork')" >&2; \
            exit 1; \
        fi; \
        git clone --filter=blob:none "${VLLM_FORK_REPO}" /opt/vllm-fork; \
        if [ -n "${VLLM_FORK_REF}" ]; then git -C /opt/vllm-fork checkout "${VLLM_FORK_REF}"; fi; \
        rm -rf /opt/vllm-fork/vllm/connex; \
        if [ "${FORK_INSTALL}" = source ]; then \
            cd /opt/vllm-fork && MAX_JOBS=${MAX_JOBS} pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} --no-build-isolation -e . ; \
        else \
            dst=$(python3 -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("vllm").origin))'); \
            echo "overlaying fork python tree onto $dst"; \
            tar -C /opt/vllm-fork/vllm --exclude=./vllm_flash_attn --exclude='*.so' -cf - . | tar -C "$dst" -xf - ; \
        fi ;; \
    *) echo "ERROR: PP_ENGINE must be upstream or fork" >&2; exit 1 ;; \
    esac

# ---- communication libraries ----------------------------------------------
RUN pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} "${UCX_PY_PKG}" "${MOONCAKE_PKG}" msgpack pyzmq

# ---- connex-rs (from the builder stage) ---------------------------------------
COPY --from=connex-rs-build /opt/ucx /opt/ucx
COPY --from=connex-rs-build /wheels /tmp/connex-rs-wheels
RUN set -eux; \
    if ls /tmp/connex-rs-wheels/*.whl >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends libibverbs1 librdmacm1 ibverbs-providers libnuma1 \
          && rm -rf /var/lib/apt/lists/*; \
        pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} /tmp/connex-rs-wheels/*.whl; \
    fi; \
    rm -rf /tmp/connex-rs-wheels

# ---- Connex ------------------------------------------------------------------
COPY . /opt/connex
# Model downloads (MODEL_PATH = Hugging Face id) land here; mount it to reuse them.
ENV HF_HOME=/root/.cache/huggingface
RUN set -eux; \
    pip install ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} --no-deps -e /opt/connex; \
    vdir=$(python3 -c 'import importlib.util,os;print(os.path.dirname(importlib.util.find_spec("vllm").origin))'); \
    if [ "${PP_ENGINE}" = fork ]; then \
        if [ "${APPLY_PATCHES}" = all ]; then \
            python3 /opt/connex/integrations/vllm/patches/apply_all.py "$vdir/v1/worker/gpu_model_runner.py"; \
        elif [ "${APPLY_PATCHES}" = block_table ]; then \
            python3 /opt/connex/integrations/vllm/patches/fast_external_pp_block_table.py "$vdir/v1/worker/gpu_model_runner.py"; \
        fi; \
        cp /opt/connex/integrations/vllm/connex_communicator.py \
           /opt/connex/integrations/vllm/connexte_communicator.py \
           "$vdir/distributed/device_communicators/"; \
    else \
        python3 -c "import vllm.engine.arg_utils as a, vllm.v1.outputs as o; \
assert hasattr(a.EngineArgs, 'pp_group_cls') and hasattr(a.EngineArgs, 'pipeline_stage_rank'), 'vLLM lacks --pp-group-cls / --pipeline-stage-rank'; \
assert 'lost_all_kv' in o.KVConnectorOutput.__dataclass_fields__, 'vLLM lacks KVConnectorOutput.lost_all_kv'; print('vLLM has the PP changes')"; \
    fi; \
    chmod +x /opt/connex/docker/*.sh; \
    cudart=$(python3 -c 'import nvidia.cuda_runtime,os;print(os.path.join(list(nvidia.cuda_runtime.__path__)[0],"lib"))' 2>/dev/null || true); \
    # No GPU driver at build time: the import check links Mooncake/UCX against
    # the CUDA toolkit's libcuda stub (only for this check; at runtime the
    # container toolkit injects the real libcuda.so.1).
    stubdir=/tmp/libcuda-stub; mkdir -p "$stubdir"; \
    stub=$(ls /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/targets/*/lib/stubs/libcuda.so 2>/dev/null | head -1 || true); \
    [ -n "$stub" ] && ln -sf "$stub" "$stubdir/libcuda.so.1"; \
    LD_LIBRARY_PATH=${cudart}:${stubdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} python3 -c "import connex, ucp, mooncake.engine; print('connex', connex.__version__, 'ucx-py', ucp.__version__)" || \
      { echo "ERROR: ucx-py/Mooncake do not load in this base image; UCX_PY_PKG (default ucx-py-cu12) and MOONCAKE_PKG must match the base image's CUDA major version" >&2; exit 1; }; \
    if [ -d /opt/ucx/lib ] && ls /opt/ucx/lib/libucp.so* >/dev/null 2>&1; then \
      LD_LIBRARY_PATH=${stubdir} python3 -c "import connex_rs; e = connex_rs.Engine('build', 0, listen_ip='127.0.0.1'); print('connex-rs', e.local_url); e.close()" || \
        { echo "ERROR: connex_rs does not load (UCX in /opt/ucx?)" >&2; exit 1; }; \
    fi; \
    rm -rf "$stubdir"

ENV CONNEX_PP_ENGINE=${PP_ENGINE}
ENTRYPOINT []
CMD ["/opt/connex/docker/launch_stage.sh"]
