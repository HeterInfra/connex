# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the vLLM ConnexConnector (needs vLLM installed; no GPU).

Drives the KVConnectorBase_V1 scheduler hooks with stand-in requests and
checks the hand-off protocol: the prefill instance holds the request's blocks
and returns kv_transfer_params; the decode instance loads the prompt tokens
asynchronously, pulls into the right local blocks (also after a resume with
generated tokens appended) and releases the hold when the request finishes.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base",
                    exc_type=ImportError)   # also skip a broken vLLM build

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

from connex.adapters.vllm.pd_connector import ConnexConnector  # noqa: E402

BS = 4


def config(kv_role, **extra):
    ktc = SimpleNamespace(kv_role=kv_role, kv_connector_extra_config={"host": "10.0.0.1", **extra},
                          kv_port=14579, engine_id="e0", is_kv_producer=kv_role == "kv_producer",
                          is_kv_consumer=kv_role == "kv_consumer")
    return SimpleNamespace(kv_transfer_config=ktc, cache_config=SimpleNamespace(block_size=BS),
                           model_config=SimpleNamespace(served_model_name="m", model="m"),
                           parallel_config=SimpleNamespace(tensor_parallel_size=1))


def request(rid, n_prompt, params, status=RequestStatus.RUNNING):
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(range(n_prompt)),
                           kv_transfer_params=params, status=status)


class Blocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self):
        return (self.ids,)


def test_prefill_holds_blocks_and_returns_params():
    c = ConnexConnector(config("kv_producer", ctrl_port=17100), KVConnectorRole.SCHEDULER, None)
    r = request("r1", 10, {"do_remote_decode": True, "transfer_id": "T"},
                RequestStatus.FINISHED_LENGTH_CAPPED)
    delay, params = c.request_finished(r, ([7, 8, 9],))
    assert delay is True
    assert params["do_remote_prefill"] and params["transfer_id"] == "T"
    assert params["remote_ctrl"] == "10.0.0.1:17100" and params["remote_num_tokens"] == 10
    meta = c.build_connector_meta(None)
    assert meta.holds == [("r1", "T", [7, 8, 9])]
    # aborted prefill: blocks are freed at once, nothing is held
    r2 = request("r2", 10, {"do_remote_decode": True, "transfer_id": "U"},
                 RequestStatus.FINISHED_ABORTED)
    assert c.request_finished(r2, ([1],)) == (False, None)
    assert c.build_connector_meta(None).holds == []


def test_decode_pulls_prompt_and_releases_on_finish():
    c = ConnexConnector(config("kv_consumer"), KVConnectorRole.SCHEDULER, None)
    p = {"do_remote_prefill": True, "transfer_id": "T", "remote_ctrl": "10.0.0.1:17100",
         "remote_num_tokens": 10}
    r = request("d1", 10, dict(p))
    assert c.get_num_new_matched_tokens(r, 0) == (10, True)
    c.update_state_after_alloc(r, Blocks([20, 21, 22]), 10)
    meta = c.build_connector_meta(None)
    (pl,) = meta.pulls
    assert (pl.ctrl, pl.transfer_id, pl.local_blocks, pl.first_block) == \
        ("10.0.0.1:17100", "T", [20, 21, 22], 0)
    assert r.kv_transfer_params["do_remote_prefill"] is False   # one transfer per scheduling
    assert c.request_finished(r, ([20, 21, 22],)) == (False, None)
    assert c.build_connector_meta(None).releases == [("10.0.0.1:17100", "T", 0)]


def test_resumed_request_loads_prompt_prefix_and_skips_local_hit():
    """After a decode instance died the router resends prompt + k generated
    tokens with the same params: only the prompt part is loaded, and blocks
    the new instance already has (prefix cache) are not pulled again."""
    c = ConnexConnector(config("kv_consumer"), KVConnectorRole.SCHEDULER, None)
    p = {"do_remote_prefill": True, "transfer_id": "T", "remote_ctrl": "h:1",
         "remote_num_tokens": 10}
    r = request("d2", 13, dict(p))                       # 10 prompt + 3 generated
    assert c.get_num_new_matched_tokens(r, 4) == (6, True)   # 1 block cached locally
    c.update_state_after_alloc(r, Blocks([5, 30, 31, 32]), 6)
    (pl,) = c.build_connector_meta(None).pulls
    assert pl.first_block == 1 and pl.local_blocks == [30, 31]


def test_release_after_pull_mode_sends_no_release():
    c = ConnexConnector(config("kv_consumer", release="after_pull"), KVConnectorRole.SCHEDULER, None)
    r = request("d3", 4, {"do_remote_prefill": True, "transfer_id": "T", "remote_ctrl": "h:1",
                          "remote_num_tokens": 4})
    c.get_num_new_matched_tokens(r, 0)
    c.update_state_after_alloc(r, Blocks([1]), 4)
    c.request_finished(r, ([1],))
    assert c.build_connector_meta(None).releases == []
