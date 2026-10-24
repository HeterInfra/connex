# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""``vllm.general_plugins`` entry point: register ``ConnexConnector`` by name.

With connex-ai installed, ``--kv-transfer-config '{"kv_connector":
"ConnexConnector", ...}'`` works without ``kv_connector_module_path``.
"""
import logging

log = logging.getLogger("connex.pd.vllm")


def register() -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    except ImportError:  # vLLM without the V1 connector factory
        return
    if "ConnexConnector" in getattr(KVConnectorFactory, "_registry", {}):
        return
    try:
        KVConnectorFactory.register_connector(
            "ConnexConnector", "connex.adapters.vllm.pd_connector", "ConnexConnector")
    except ValueError:  # registered concurrently / by vLLM itself
        pass
