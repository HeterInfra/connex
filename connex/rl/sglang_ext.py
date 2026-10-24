# SPDX-License-Identifier: Apache-2.0
"""Connex weight sync for SGLang, through SGLang's custom weight loader hook
(no SGLang changes).

Start the server with::

    --custom-weight-loader connex.rl.sglang_ext.connex_load

and trigger an update with ``update_weights_from_tensor`` whose payload is
a single control tensor built by ``control_payload`` (registry URL, version)
and ``load_format="connex.rl.sglang_ext.connex_load"``. In every TP
scheduler process SGLang then calls ``connex_load(model, named_tensors)``
between scheduler steps; it pulls the version with a ``WeightSubscriber``
(one per process, created on first use) and calls ``model.load_weights``.

The HTTP form (``POST /update_weights_from_tensor``) needs the payload
serialized per TP rank; ``connex.rl.engines.SGLangHTTP`` does that.
"""
import json
import os
import socket

import torch

from .registry import RegistryClient
from .sync import DEFAULT_CHUNK, WeightSubscriber
from .transport import advertise_ip, make_transport

LOADER = "connex.rl.sglang_ext.connex_load"
CONTROL = "__connex_weight_sync__"

_sub = None
last_stats = {}


def control_payload(registry_url, version=None, member=None, transport="connex-rs",
                    chunk_bytes=DEFAULT_CHUNK):
    d = {"registry_url": registry_url, "version": version, "member": member,
         "transport": transport, "chunk_bytes": chunk_bytes}
    return [(CONTROL, torch.frombuffer(bytearray(json.dumps(d).encode()), dtype=torch.uint8))]


def parse_control(named_tensors):
    for name, t in named_tensors:
        if name == CONTROL:
            return json.loads(bytes(t.cpu().tolist()).decode())
    return None


def _subscriber(ctl, device):
    global _sub
    if _sub is None:
        rank = int(os.environ.get("CONNEX_TP_RANK", torch.cuda.current_device()
                                  if torch.cuda.is_available() else 0))
        base = ctl.get("member") or f"{socket.gethostname()}-{os.getpid()}"
        member = f"{base}/r{rank}"
        if ctl.get("transport") == "local":
            t = make_transport("local", name=member)
        else:
            t = make_transport("connex-rs", lid=("connex-rl", 1, rank, os.getpid()),
                               advertise_ip=advertise_ip())
        _sub = WeightSubscriber(RegistryClient(ctl["registry_url"]), t, member,
                                device=device, chunk_bytes=ctl.get("chunk_bytes", DEFAULT_CHUNK))
    return _sub


def connex_load(model, named_tensors):
    """SGLang custom weight loader: ``named_tensors`` carries the control
    record, the weights come over connex-rs."""
    global last_stats
    ctl = parse_control(named_tensors)
    if ctl is None:                       # plain tensors: behave like the default
        model.load_weights(named_tensors)
        return
    if os.environ.get("CONNEX_RL_DEVICE"):          # tests: force a device
        dev = torch.device(os.environ["CONNEX_RL_DEVICE"])
    elif torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = torch.device("cpu")
    sub = _subscriber(ctl, dev)
    m = sub.pull(ctl.get("version"))
    model.load_weights(list(sub.tensors()))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sub.commit(m.version)
    last_stats = dict(sub.stats)
