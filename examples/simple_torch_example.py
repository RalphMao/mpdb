import torch.distributed as dist

dist.init_process_group('nccl')

from mpdb.shell import embed

embed()
