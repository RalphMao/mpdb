import os
import datetime

def get_local_rank():
    # Check for torchrun first
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    
    # Check for MPI environment variables
    for env_var in ['OMPI_COMM_WORLD_LOCAL_RANK', 'MPI_LOCALRANKID', 'PMI_LOCAL_RANK']:
        rank = os.environ.get(env_var)
        if rank is not None:
            return int(rank)
    
    # Default to 0 if no rank found
    return 0

def get_dist_backend():
    dist = get_pytorch_dist()
    if dist is not None:
        return dist
    return DummyBackend()
 
class DummyBackend:
    is_dummy = True

    def get(self):
        return 0

    def set(self, value):
        pass

    def sync(self):
        pass
    
    def finish(self):
        pass

class TorchDistBackend(DummyBackend):
    is_dummy = False

    def __init__(self):
        import torch.distributed as dist
        assert dist.is_initialized()
        self.dist = dist
        rank = dist.get_rank()
        if rank == 0:
            self.store = dist.TCPStore("127.0.0.1", 50001, is_master=True)
            self.store.set("active", "0")
        # dist.barrier()
        if rank != 0:
            self.store = dist.TCPStore("127.0.0.1", 50001, is_master=False, timeout=datetime.timedelta(seconds=30))

    def get(self):
        return int(self.store.get("active"))

    def set(self, value: int):
        self.store.set("active", str(value))

    def sync(self):
        self.dist.barrier()

    def finish(self):
        self.store.set("active", "-1")

def get_pytorch_dist():
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if not dist.is_initialized():
        print("PyTorch distributed is not initialized, disable")
        return None
    return TorchDistBackend()
