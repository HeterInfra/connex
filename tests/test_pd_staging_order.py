"""Host staging (issue #3): stage_in lands after work already queued on the
engine stream that writes the same blocks (GPU only)."""
import threading

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_stage_in_is_ordered_after_queued_engine_writes():
    from connex.pd.host_staging import HostStaging
    kv = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
    hs = HostStaging([(kv.data_ptr(), kv.numel())])
    m = hs._spans[0][3]
    m.fill_(7)                                     # the pulled KV, in the mirror
    torch.cuda._sleep(int(0.2 * 1.4e9))            # engine busy ~200 ms ...
    kv.fill_(1)                                    # ... then a stale step writes the block
    t = threading.Thread(target=hs.stage_in, args=([(kv.data_ptr(), kv.numel())],))
    t.start(); t.join()                            # the pull thread stages the KV in
    torch.cuda.synchronize()
    assert int(kv[0]) == 7 and int(kv[-1]) == 7
