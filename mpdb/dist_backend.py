import datetime
import os
import time
import typing


def get_local_rank():
    # Check for torchrun/accelerate, OpenMPI, Intel MPI, Slurm
    for env_var in ["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_RANK", "SLURM_LOCALID"]:
        rank = os.environ.get(env_var)
        if rank is not None:
            return int(rank)

    # Default to 0 if no rank found
    return 0


def get_local_world_size():
    # Check for torchrun/accelerate, OpenMPI, Intel MPI, Slurm
    for env_var in [
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS_PER_NODE",
    ]:
        rank = os.environ.get(env_var)
        if rank is not None:
            return int(rank)

    # Default to 1 if no distributed setup is detected
    return 1


def get_dist_backend():
    dist = get_pytorch_dist()
    if dist is not None:
        dist.set(0)
        return dist
    return DummyBackend()


def get_pytorch_dist():
    try:
        import torch.distributed as dist
    except ImportError:
        print("Process switching relies on torch.distributed.TCPStore")
        print("torch not found, disable process switching")
        return None
    return TorchDistBackend()


class DummyBackend:
    is_dummy = True

    def get(self) -> int:
        return 0

    def set(self, value: int):
        pass

    def sync(self):
        pass

    def finish(self):
        pass


class Singleton(type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TorchDistBackend(DummyBackend, metaclass=Singleton):
    is_dummy = False

    def __init__(self):
        import torch.distributed as dist

        self.dist = dist
        self.rank = get_local_rank()
        if self.rank == 0:
            self.store = dist.TCPStore("127.0.0.1", 50001, is_master=True)
            self.store.set("active", "0")
        else:
            self.store = dist.TCPStore(
                "127.0.0.1",
                50001,
                is_master=False,
                timeout=datetime.timedelta(seconds=30),
            )

    def get(self) -> int:
        return int(self.store.get("active"))

    def set(self, value: int):
        self.store.set("active", str(value))

    def sync(self):
        self.dist.barrier()

    def finish(self):
        self.store.set("active", "-1")
        if self.rank == 0:
            n_sec = 2
            print(
                f"Sending STOP message to other sessions. Exiting in {n_sec} seconds.",
            )
            time.sleep(n_sec)
