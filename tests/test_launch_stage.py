# SPDX-License-Identifier: Apache-2.0
"""docker/launch_stage.sh GPU selection, with fake nvidia-smi / python3."""
import os
import stat
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "docker", "launch_stage.sh")


def _exe(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def fakebin(tmp_path):
    b = tmp_path / "bin"
    b.mkdir()
    # 4 GPUs: 0 and 3 busy, 1 and 2 free
    _exe(b / "nvidia-smi", 'case "$*" in *topo*) exit 0;; esac\n'
         'rows="0, 1000\n1, 45000\n2, 45000\n3, 2000"\n'
         'if [ "$1" = -i ]; then echo "$rows" | awk -F", " -v i="$2" \'$1==i{print $2}\'; exit 0; fi\n'
         'echo "$rows"\n')
    _exe(b / "python3", 'for a in "$@"; do [ "$a" = vllm.entrypoints.openai.api_server ] && '
         '{ echo "CVD=$CUDA_VISIBLE_DEVICES"; exit 0; }; done; exit 0\n')
    return b


def _pick(fakebin, **env):
    e = {"PATH": f"{fakebin}:/usr/bin:/bin", "ROUTER_ADDR": "r:1", "NUM_STAGES": "4",
         "GPU_SELECT": "auto", "HOST_IP": "127.0.0.1", **{k: str(v) for k, v in env.items()}}
    out = subprocess.run(["bash", SCRIPT], env=e, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return [l for l in out.stdout.splitlines() if l.startswith("CVD=")][0][4:]


def test_auto_select_spreads_stages_that_start_together(fakebin):
    # two stages of one node see the same two free GPUs and must not collide
    picks = {_pick(fakebin, STAGE=s) for s in (0, 1)}
    assert picks == {"1", "2"}
    assert {_pick(fakebin, STAGE=s) for s in (2, 3)} == {"1", "2"}
    assert {_pick(fakebin, STAGE=1, REPLICA=r) for r in (0, 1)} <= {"1", "2"}


def test_spread_is_stable_after_a_sibling_started_allocating(fakebin, tmp_path):
    # stage 0 already holds 2 GB on GPU 1 when stage 1 looks: it must still
    # take the other free GPU, not the one with the most free memory order
    _exe(fakebin / "nvidia-smi", 'case "$*" in *topo*) exit 0;; esac\n'
         'printf "0, 1000\\n1, 43000\\n2, 45000\\n3, 2000\\n"\n')
    assert _pick(fakebin, STAGE=0) == "1" and _pick(fakebin, STAGE=1) == "2"


def test_busy_gpus_are_never_picked(fakebin):
    for s in range(4):
        assert _pick(fakebin, STAGE=s, MIN_FREE_MIB=30000) in ("1", "2")


def test_preferred_gpu_wins_when_free(fakebin):
    assert _pick(fakebin, STAGE=0, PREFERRED_GPU=2) == "2"
