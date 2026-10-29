# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Connex project
"""Regression tests for review fixes in connex.adapters.vllm.pp (CPU only)."""

import os
import pickle
import queue
import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from connex.adapters.vllm.pp import link as L  # noqa: E402


class _Evil:
    def __reduce__(self):
        return (os.system, ("touch /nonexistent-connex-review-marker",))


def test_data_plane_refuses_pickled_globals():
    with pytest.raises(pickle.UnpicklingError):
        L.loads_plain(pickle.dumps((L.ABORT, _Evil())))
    # plain messages still round-trip
    msg = (L.TENSORS, (1, 2, -1), [("h", "torch.bfloat16", (4, 8), 0, 64)], 64, None,
           ("ipc", ("cxpp-1-h-1", 3, 4096, (0, b"\x00" * 64, 4096, 0, b"", 0, b"", True)), 0))
    assert L.loads_plain(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)) == msg


def _bare_link():
    lk = object.__new__(L.StageLink)
    lk.inbox = L._Inbox()
    lk.push_fallback = {}
    lk.min_epoch = 0
    lk._closed = False
    return lk


def test_handle_drops_malicious_and_malformed_messages():
    lk = _bare_link()
    called = []
    lk.abort = called.append
    lk._handle(("p",), pickle.dumps((L.ABORT, _Evil())))   # refused, not executed
    lk._handle(("p",), pickle.dumps(5))                     # not a tuple
    lk._handle(("p",), b"garbage")
    assert called == []
    lk._handle(("p",), pickle.dumps((L.ABORT, 3)))
    assert called == [3]


def test_rx_loop_survives_a_handler_error():
    lk = _bare_link()
    msgs = queue.Queue()
    for m in [(("p",), pickle.dumps((L.BYTES,))),          # unpack error inside _handle
              (("p",), pickle.dumps((L.BYTES, (0, 1, 0), "u", b"x")))]:
        msgs.put(m)

    def recv_msg(timeout_ms):
        try:
            return msgs.get_nowait()
        except queue.Empty:
            lk._closed = True
            return None

    lk.engine = SimpleNamespace(recv_msg=recv_msg)
    t = threading.Thread(target=lk._rx_loop)
    t.start()
    t.join(5)
    assert not t.is_alive()
    assert lk.inbox.items[(0, 1, 0)] == b"x"   # the second message was still handled


def test_chain_of_an_aborted_epoch_starts_cancelled():
    import numpy as np

    from connex.adapters.vllm.pp.rdmachan import PushTx

    tx = object.__new__(PushTx)
    tx.cancel_np = np.zeros(64, dtype=np.int32)
    tx.cancel = SimpleNamespace(data_ptr=lambda: 1000)
    tx.word_epoch = [-1] * 64
    tx.dead = False
    tx.cancel_below(5)          # no word of epoch 3 existed yet
    tx._word(3)
    assert tx.cancel_np[3] == 1
    tx._word(5)
    assert tx.cancel_np[5] == 0

