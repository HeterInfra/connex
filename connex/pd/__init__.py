# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Prefill-decode disaggregation: engine-neutral KV transfer and PD pools.

* ``connex.pd.kv_layout``: where a KV block's bytes live; block runs to reads.
* ``connex.pd.dataplane``: connex-rs / Mooncake / loopback data planes.
* ``connex.pd.transfer``: ``KVProducer`` (prefill holds KV as a renewable
  lease) and ``KVConsumer`` (decode pulls it, possibly again after a decode
  instance died).
* ``connex.pd.membership``: PD pools in the Connex front-router.

Engine adapters: ``connex.adapters.vllm.pd_connector`` (vLLM KV connector),
``connex.adapters.sglang.pd_backend`` (SGLang transfer backend). Router:
``connex.serving.pd_router``. Nothing here imports an inference engine.
"""
