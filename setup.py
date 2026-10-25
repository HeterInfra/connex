# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
import os
import re

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))


def _version():
    with open(os.path.join(HERE, "connex", "__init__.py")) as f:
        return re.search(r'^__version__\s*=\s*"([^"]+)"', f.read(), re.M).group(1)


with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="connex-ai",  # PyPI distribution; the import name is `connex`
    version=_version(),
    description="Communication library for distributed AI infrastructure: endpoint mobility primitives that let replicas join, leave, fail over and migrate with traffic in flight (PD/AF disaggregation, pipeline parallelism, elastic serving)",
    keywords=["llm serving", "distributed inference", "endpoint mobility", "pipeline parallelism",
              "prefill-decode disaggregation", "attention-ffn disaggregation", "multi-replica", "elastic serving", "heterogeneous inference", "failover", "rdma", "mooncake", "ucx", "vllm"],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Connex Contributors",
    url="https://github.com/HeterInfra/connex",
    project_urls={
        "Source": "https://github.com/HeterInfra/connex",
        "Changelog": "https://github.com/HeterInfra/connex/blob/main/CHANGELOG.md",
        "connex-rs": "https://github.com/HeterInfra/connex-rs",
    },
    license="Apache-2.0",
    packages=find_packages(include=["connex", "connex.*"]),
    python_requires=">=3.10",
    install_requires=["torch", "pyzmq", "msgpack", "numpy", "safetensors"],
    extras_require={
        "ucx": ["ucx-py-cu12"],
        # connex-rs data plane (Rust over UCX), its own repository and package
        # (https://github.com/HeterInfra/connex-rs). For RDMA/GPUDirect, build
        # the wheel against your UCX (third_party/connex-rs submodule, maturin)
        # or use the image (docker/Dockerfile).
        "rs": ["connex-rs>=0.1.1,<0.2"],
        "mooncake": ["mooncake-transfer-engine"],
        # Request retry/resume proxy (connex.serving.proxy) and benchmarks.
        "proxy": ["aiohttp"],
        "bench": ["aiohttp"],
        # Engine adapters (connex.adapters.*). PD disaggregation works with
        # upstream vLLM (KV connector) and SGLang (transfer backend plugin);
        # the pipeline-parallel adapter needs the Connex vLLM fork.
        "vllm": ["vllm", "aiohttp"],
        "sglang": ["sglang", "aiohttp"],
        "test": ["pytest", "pyflakes", "aiohttp"],
        "dev": ["pytest", "pyflakes", "aiohttp", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "connex-front-router=connex.front_router:main",
            "connex-directory=connex.directory_server:main",
            "connex-pd-router=connex.serving.pd_router:main",
        ],
        # Upstream engines, no fork: vLLM finds the ConnexConnector by name and
        # the "connex" weight-transfer backend; SGLang gets the "connex"
        # disaggregation transfer backend and the connex.ep control path (CONNEX_EP=1).
        "vllm.general_plugins": [
            "connex_pd = connex.adapters.vllm.plugin:register",
            "connex_rl = connex.rl.vllm_plugin:register",
            "connex_ep = connex.ep.vllm_ext:register",
        ],
        "sglang.srt.plugins": [
            "connex = connex.adapters.sglang.pd_backend:register",
            "connex_ep = connex.ep.sglang_ext:register",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Distributed Computing",
    ],
)
