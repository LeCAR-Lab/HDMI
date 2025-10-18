import torch
import inspect
import abc
import weakref
from typing import Tuple, TYPE_CHECKING, Generic, TypeVar
from collections import defaultdict
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_mul


if TYPE_CHECKING:
    from active_adaptation.envs.base import _Env


def sample_quat_yaw(size, yaw_range=(0, torch.pi * 2), device: torch.device = "cpu"):
    yaw = torch.rand(size, device=device).uniform_(*yaw_range)
    quat = torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )
    return quat


class Command:
    def __init__(self, env, teleop: bool = False) -> None:
        self.env: _Env = env
        self.asset = self.env.scene.entities["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()
        self.teleop = teleop

        if hasattr(self.env.scene, "terrain"):
            self.terrain_type = self.env.scene.terrain.cfg.terrain_type
        else:
            self.terrain_type = "plane"

        if self.terrain_type == "generator":
            self._origins = self.env.scene.terrain.terrain_origins.reshape(
                -1, 3
            ).clone()

        if self.teleop:
            # acquire omniverse interfaces
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                    event, *args
                ),
            )
            self.key_pressed = defaultdict(lambda: False)

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Called before `reset` to sample initial state for the next episodes.
        This can be used for implementing curriculum learning.
        """
        init_root_state = self.init_root_state[env_ids]
        if self.terrain_type == "plane":
            origins = self.env.scene.env_origins[env_ids]
        else:
            idx = torch.randint(
                0, len(self._origins), (len(env_ids),), device=self.device
            )
            origins = self._origins[idx]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7], sample_quat_yaw(len(env_ids), device=self.device)
        )
        return init_root_state

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.key_pressed[event.input.name] = True
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.key_pressed[event.input.name] = False


class _RegistryMixin:

    def __init_subclass__(cls) -> None:
        """Put the subclass in the global registry"""
        if not hasattr(cls, "registry"):
            cls.registry = {}

        cls_name = cls.__name__
        try:
            cls._file = inspect.getfile(cls)
            cls._line = inspect.getsourcelines(cls)[1]
        except:
            cls._file = "unknown"
            cls._line = "unknown"

        if cls_name.startswith("_"):
            return
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")


CT = TypeVar("CT", bound=Command)


class Observation(Generic[CT], _RegistryMixin):
    """
    Base class for all observations.
    """

    def __init__(self, env):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor

    def startup(self):
        """Called once upon initialization of the environment"""
        pass

    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass


class Reward(Generic[CT], _RegistryMixin):
    def __init__(
        self,
        env,
        weight: float,
        enabled: bool = True,
    ):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager
        self.weight = weight
        self.enabled = enabled

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def step(self, substep: int):
        pass

    def post_step(self, substep: int):
        pass

    def update(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def __call__(self) -> torch.Tensor:
        result = self.compute()
        if isinstance(result, torch.Tensor):
            rew, count = result, result.numel()
        elif isinstance(result, tuple):
            rew, is_active = result
            rew = rew * is_active.float()
            count = is_active.sum().item()
        return self.weight * rew, count

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError

    def debug_draw(self):
        pass


class Randomization(Generic[CT], _RegistryMixin):
    def __init__(self, env):
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def startup(self):
        pass

    def reset(self, env_ids: torch.Tensor):
        pass

    def step(self, substep):
        pass

    def update(self):
        pass

    def debug_draw(self):
        pass


class Termination(Generic[CT], _RegistryMixin):
    def __init__(self, env, **kwargs):
        if kwargs:
            print("Warning: Unused kwargs in Termination:", kwargs)
            breakpoint()
        super().__init__(**kwargs)
        self.env: _Env = env
        self.command_manager: CT = env.command_manager

    def update(self):
        pass

    def reset(self, env_ids):
        pass

    @abc.abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device
