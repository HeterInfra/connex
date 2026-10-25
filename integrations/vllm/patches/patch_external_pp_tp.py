# SPDX-License-Identifier: Apache-2.0
"""Patch 16 (tensor parallelism per stage): an external-PP stage may run with
``--tensor-parallel-size N``; the TP group is one pipeline endpoint.

    python patch_external_pp_tp.py /workspace/vllm/vllm/v1/worker/gpu_model_runner.py

Edits ../../config/parallel.py and ../../distributed/parallel_state.py
relative to the runner path (the runner itself is not modified). Idempotent.

Design
------
A stage with TP size N runs N worker processes (vLLM's multiprocess
executor), all executing the same steps in lockstep (the scheduler of stage 0
broadcasts each step to its workers; on stages 1+ every worker runs the
recv -> compute -> send loop thread). Only TP rank 0 owns a Connex pipeline
endpoint: it registers the stage's logical endpoint (model, stage, replica)
with the front-router, receives every step from the upstream stage and sends
every reply. The other ranks never touch the network between stages:

* ``recv_tensor_dict``: rank 0 receives from Connex and broadcasts the dict
  inside the TP group (``GroupCoordinator.broadcast_tensor_dict``: metadata
  over the CPU group, tensors over NCCL/NVLink); a timeout (``None``) is
  broadcast too, so every rank returns None and retries or idles together.
* ``send_tensor_dict``: rank 0 sends; the others drop the call. Hidden states
  and residual leave each layer replicated on every TP rank (after the
  all-reduce), and stage 0's sampled tokens are identical on all ranks, so
  rank 0's copy is the stage's output.

Failure handling is group-level (connex.pp.group): rank 0's endpoint
registers as shard 0 of a group of N, the other ranks register shards 1..N-1
for liveness (TPShardMember). If any rank dies, its lease lapses and the
front-router fails the whole group: all shards are deregistered and the
stage's peers hand its in-flight steps over to the same stage of another
replica chain, without waiting for rank 0 (which may hang in NCCL rather than
exit). Peers only route to complete groups; the stage rejoins under the same
logical endpoint when it is restarted.

Limits: CONNEX_PP_INFLIGHT must be 1 on a TP>1 stage (the pipelined loops
call the communicator outside the group wrapper); a ValueError says so at
start-up. Re-prefill ids reach the scheduler from rank 0's output (worker
outputs are pickled, so the extra attribute survives).
"""
import os
import sys

MARK = "# connex: tensor-parallel external stage"

CONFIG_OLD = """            # Enforce current limitation: only TP=1, DP=1 are supported.
            if self.tensor_parallel_size != 1:
                raise ValueError(
                    "tensor_parallel_size must be 1 when pipeline_stage_mode is "
                    "'external'. Tensor parallel external pipeline is not yet "
                    "supported."
                )
"""
CONFIG_NEW = f"""            {MARK}: TP > 1 is supported (TP rank 0 is
            # the stage's pipeline endpoint, the group shares steps over NCCL);
            # DP > 1 is not.
            if (self.tensor_parallel_size != 1
                    and int(os.environ.get("CONNEX_PP_INFLIGHT", "1") or "1") > 1):
                raise ValueError(
                    "tensor_parallel_size > 1 in external pipeline mode needs "
                    "CONNEX_PP_INFLIGHT=1."
                )
"""

PS_SEND_OLD = """        def send_tensor_dict(self, tensor_dict, dst=None, **kwargs):
            if self.zeromq_comm is not None:
                self.zeromq_comm.send_tensor_dict(tensor_dict, dst)
                return None
            raise RuntimeError("ZeroMQ communicator not initialized")
"""
PS_SEND_NEW = f"""        {MARK}: only TP rank 0 talks to other stages
        tp_size = 1
        tp_rank = 0

        def send_tensor_dict(self, tensor_dict, dst=None, **kwargs):
            if self.tp_rank != 0:
                return None   # rank 0 sends the stage's (replicated) output
            if self.zeromq_comm is not None:
                self.zeromq_comm.send_tensor_dict(tensor_dict, dst)
                return None
            raise RuntimeError("ZeroMQ communicator not initialized")
"""
PS_RECV_OLD = """        def recv_tensor_dict(self, src=None, **kwargs):
            if self.zeromq_comm is not None:
                return self.zeromq_comm.recv_tensor_dict(src)
            raise RuntimeError("ZeroMQ communicator not initialized")
"""
PS_RECV_NEW = """        def recv_tensor_dict(self, src=None, **kwargs):
            if self.tp_size > 1:
                tp = get_tp_group()
                # called from loop / executor threads, whose current CUDA
                # device is 0: received tensors must land on this rank's GPU
                if tp.device.type == "cuda":
                    torch.cuda.set_device(tp.device)
                if self.tp_rank == 0:
                    td = self.zeromq_comm.recv_tensor_dict(src)
                    tp.broadcast_tensor_dict(
                        td if td is not None else {"__cx_none__": True}, src=0)
                    return td
                td = tp.broadcast_tensor_dict(None, src=0)
                return None if td is None or td.get("__cx_none__") else td
            if self.zeromq_comm is not None:
                return self.zeromq_comm.recv_tensor_dict(src)
            raise RuntimeError("ZeroMQ communicator not initialized")
"""

PS_COMM_OLD = """    # Initialize communicator based on backend
    pp_comm = CommunicatorClass(
        stage_idx=stage_idx,
        num_stages=num_stages,
        router_port=router_port,
        front_router_addr=front_router_addr,
        device=device,
    )
    pp_group.zeromq_comm = pp_comm  # Keep name for compatibility
"""
PS_COMM_NEW = f"""    {MARK}: one pipeline endpoint per TP group (rank 0)
    _tp = get_tp_group()
    pp_group.tp_size = _tp.world_size
    pp_group.tp_rank = _tp.rank_in_group
    if pp_group.tp_size > 1:
        logger.info("External PP stage %d: TP rank %d of %d%s", stage_idx,
                    pp_group.tp_rank, pp_group.tp_size,
                    " (pipeline endpoint)" if pp_group.tp_rank == 0 else "")
    if pp_group.tp_size > 1:
        # the stage's endpoint registers as shard 0 of a group of tp_size
        os.environ["CONNEX_PP_TP_SIZE"] = str(pp_group.tp_size)
    if pp_group.tp_rank > 0:
        # other ranks register as the group's other shards, for liveness: a
        # dead rank fails the whole stage endpoint at the front-router
        try:
            from connex.pp.group import TPShardMember

            pp_group.tp_member = TPShardMember(stage_idx, pp_group.tp_rank, pp_group.tp_size,
                                               front_router_api=front_router_addr)
            pp_group.tp_member.start()
        except Exception as e:  # noqa: BLE001
            logger.warning("External PP stage %d: TP shard registration failed: %s",
                           stage_idx, e)
    if pp_group.tp_rank == 0:
        # Initialize communicator based on backend
        pp_comm = CommunicatorClass(
            stage_idx=stage_idx,
            num_stages=num_stages,
            router_port=router_port,
            front_router_addr=front_router_addr,
            device=device,
        )
        pp_group.zeromq_comm = pp_comm  # Keep name for compatibility
"""


def _edit(path, edits, need_os=False):
    src = open(path).read()
    if MARK in src:
        return "already patched"
    for old, new in edits:
        if src.count(old) != 1:
            raise SystemExit(f"{os.path.basename(path)}: anchor not found/unique: "
                             f"{old.strip()[:70]!r}")
        src = src.replace(old, new)
    if need_os and "\nimport os\n" not in src:
        src = "import os\n" + src
    open(path, "w").write(src)
    return "patched"


def main(argv):
    runner = os.path.abspath(argv[1])
    root = os.path.dirname(os.path.dirname(os.path.dirname(runner)))   # .../vllm
    cfg = os.path.join(root, "config", "parallel.py")
    ps = os.path.join(root, "distributed", "parallel_state.py")
    r1 = _edit(cfg, [(CONFIG_OLD, CONFIG_NEW)], need_os=True)
    r2 = _edit(ps, [(PS_SEND_OLD, PS_SEND_NEW), (PS_RECV_OLD, PS_RECV_NEW),
                  (PS_COMM_OLD, PS_COMM_NEW)], need_os=True)
    print(f"config/parallel.py: {r1}; distributed/parallel_state.py: {r2}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
