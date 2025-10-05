import torch
import weakref

from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_mul
from typing import Sequence, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
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
    def __init__(self, env, teleop: bool=False) -> None:
        self.env: _Env = env
        self.asset: Articulation = env.scene["robot"]
        self.init_root_state = self.asset.data.default_root_state.clone()
        self.init_root_state[:, 3:7] = self.asset.data.root_state_w[:, 3:7]
        self.init_joint_pos = self.asset.data.default_joint_pos.clone()
        self.init_joint_vel = self.asset.data.default_joint_vel.clone()

        if hasattr(self.env.scene, "terrain"):
            self.terrain_type = self.env.scene.terrain.cfg.terrain_type
        else:
            self.terrain_type = "plane"
        
        if self.terrain_type == "generator":
            self._origins = self.env.scene.terrain.terrain_origins.reshape(-1, 3).clone()

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
            idx = torch.randint(0, len(self._origins), (len(env_ids),), device=self.device)
            origins = self._origins[idx]
        init_root_state[:, :3] += origins
        init_root_state[:, 3:7] = quat_mul(
            init_root_state[:, 3:7],
            sample_quat_yaw(len(env_ids), device=self.device)
        )
        return init_root_state

