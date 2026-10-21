# SPDX-License-Identifier: Apache-2.0
"""Locate the external-PP vLLM fork source used by the patch tests.

Set VLLM_FORK_DIR to the root of a checkout of the fork (the directory that
contains the ``vllm/`` package). Tests that need it skip when it is unset or
missing.
"""
import os

FORK_DIR = os.environ.get("VLLM_FORK_DIR", "")
FORK_V1 = os.path.join(FORK_DIR, "vllm", "v1") if FORK_DIR else "/nonexistent/vllm/v1"
