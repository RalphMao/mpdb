"""
Test this file by:
    torchrun --nproc-per-node 4 examples/simple_torch_example.py

Alternatively, you can also test with accelerate:
    accelerate launch --multi_gpu --num-processes 4 examples/simple_torch_example.py
"""
import torch.distributed as dist

from mpdb import embed

embed()
dist.init_process_group("nccl")
embed()
