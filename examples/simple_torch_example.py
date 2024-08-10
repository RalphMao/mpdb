from __future__ import annotations

import torch.distributed as dist

from mpdb.shell import embed

dist.init_process_group('nccl')
embed()
