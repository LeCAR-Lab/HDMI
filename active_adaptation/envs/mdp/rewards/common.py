from active_adaptation.envs.mdp.base import Reward

import torch
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from isaaclab.sensors import ContactSensor
    
class survival(Reward):
    def compute(self):
        return torch.ones(self.num_envs, 1, device=self.device)

class linvel_z_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        linvel_z = self.asset.data.root_lin_vel_b[:, 2].unsqueeze(1)
        return -linvel_z.square()

class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_names: str = None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None

    def update(self):
        if self.body_ids is not None:
            angvel = self.asset.data.body_link_ang_vel_w[:, self.body_ids]
        else:
            angvel = self.asset.data.root_com_ang_vel_w.unsqueeze(1)
        self.angvel_w = angvel

    def compute(self) -> torch.Tensor:
        r = -self.angvel_w[:, :, :2].square().sum(-1).mean(1)
        return r.reshape(self.num_envs, 1).clamp_min(-1.0)

class body_upright(Reward):
    """
    Reward for keeping the specified body upright.
    """
    def __init__(self, env, body_name: str, weight, enabled = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id, body_name = self.asset.find_bodies(body_name)
        self.down = torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, len(self.body_id), 3)
    
    def compute(self) -> torch.Tensor:
        g = quat_apply_inverse(
            self.asset.data.body_link_quat_w[:, self.body_id],
            self.down
        )
        rew = 1. - g[:, :, :2].square().sum(-1)
        return rew.mean(1, True)

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True)

class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        limits = self.asset.data.joint_effort_limits[:, self.joint_ids] * soft_factor
        self.soft_limits = limits
    
    def compute(self) -> torch.Tensor:
        applied_torque = self.asset.data.actuator_force[:, self.joint_ids]
        violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        return - (violation_high + violation_low).sum(dim=1, keepdim=True)

class action_rate_l2(Reward):
    """Penalize the rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew

class action_rate2_l2(Reward):
    """Penalize the second order rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = (
            action_buf[:, :, 0] - 2 * action_buf[:, :, 1] + action_buf[:, :, 2]
        )
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew


class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().clamp_max(5.0).sum(1, True)


class undesired_contact_force_xy(Reward):
    def __init__(self, body_names: str | List[str], thres: float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_ids = self.contact_forces.find_bodies(body_names)[0]
        self.thres = thres
    
    def compute(self):
        contact_forces = self.contact_forces.data.net_forces_w[:, self.feet_ids]
        contact_forces = (contact_forces[:, :, :2].norm(dim=-1) - self.thres).clamp_min(0.0)
        return - contact_forces.mean(dim=1, keepdim=True)
    
