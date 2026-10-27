# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Connex pipeline parallelism on upstream vLLM (no fork, no patches).

* ``group.ConnexPPGroup``      ``--pp-group-cls``: PP traffic over connex-rs
* ``worker.ConnexPPWorker``    ``--worker-cls``: hands each step's route to the group
* ``executor.ConnexPPExecutor`` ``--distributed-executor-backend``: stage 0
  drives separate stage instances with replicas and in-flight failover
* ``stage``                    ``python -m connex.adapters.vllm.pp.stage``: a
  downstream stage instance

Needs a vLLM with ``--pp-group-cls`` (in-process mode), plus
``--pipeline-stage-rank`` and ``KVConnectorOutput.lost_kv_req_ids``
(separate instances with failover). See integrations/vllm/README.md.
"""
