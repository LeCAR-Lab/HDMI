import os
import active_adaptation.learning

_BACKEND = "mjlab"

_LOCAL_RANK = os.getenv("LOCAL_RANK", "0")
_LOCAL_RANK = int(_LOCAL_RANK)
_WORLD_SIZE = os.getenv("WORLD_SIZE", "1")
_WORLD_SIZE = int(_WORLD_SIZE)
_MAIN_PROCESS = _LOCAL_RANK == 0


def is_main_process():
    return _MAIN_PROCESS

def is_distributed():
    return _WORLD_SIZE > 1

def get_local_rank():
    return _LOCAL_RANK

def get_world_size():
    return _WORLD_SIZE

_print = print
def print(*args, **kwargs):
    _print(f"[RANK {_LOCAL_RANK}/{_WORLD_SIZE}]:", *args, **kwargs)


ASSET_PATH = os.path.join(os.path.dirname(__file__), "assets")

def set_backend(backend: str):
    if not backend in ("isaac", "mujoco"):
        raise ValueError(f"backend must be either 'isaac' or 'mujoco', got {backend}")
    global _BACKEND
    _BACKEND = backend


def get_backend():
    return _BACKEND

