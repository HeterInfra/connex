# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter (external pipeline mode of the vLLM fork).

``connex.adapters.vllm.connexte.ConnexTECommunicator`` is the communicator the
fork instantiates for ``--pipeline-communicator connexte``. The fork loads it
as ``vllm.distributed.device_communicators.connexte_communicator``; the file
integrations/vllm/connexte_communicator.py copied there is a re-export shim.
Engine-side changes live in integrations/vllm/patches.
"""
