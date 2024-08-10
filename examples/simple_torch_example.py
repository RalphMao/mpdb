import torch.distributed as dist

from mpdb import embed

dist.init_process_group("nccl")
embed()
