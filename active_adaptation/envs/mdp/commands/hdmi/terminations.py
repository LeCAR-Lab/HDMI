from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking, RobotObjectTracking
from active_adaptation.envs.mdp.base import Termination as BaseTermination

import torch
from typing import List
from omegaconf import ListConfig
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate, axis_angle_from_quat
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

class _cum_error_mixin:
    def __init__(self, min_steps: int=1, threshold: float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.min_steps = min_steps
        self.threshold = threshold

        with torch.device(self.device):
            self.error = torch.zeros(self.num_envs)
            self.__exceeded = torch.zeros(self.num_envs, dtype=bool)
            self.__cum_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        
    def update(self):
        self.__exceeded = self.error >= self.threshold
        self.__cum_steps[self.__exceeded] += 1
        self.__cum_steps[~self.__exceeded] = 0

    def reset(self, env_ids):
        self.__cum_steps[env_ids] = 0
    
    def __call__(self):
        return (self.__cum_steps >= self.min_steps).unsqueeze(-1)
        
RobotTrackTermination = BaseTermination[RobotTracking]
class cum_body_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_link_pos_w - robot_body_link_pos_w).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()

class cum_body_z_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_link_pos_w - robot_body_link_pos_w)[..., 2].abs()
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()
    
class cum_body_ori_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_link_quat_w = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]
        robot_body_link_quat_w = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(quat_conjugate(ref_body_link_quat_w), robot_body_link_quat_w)
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()
    
class cum_body_pos_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_link_pos_w = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        ref_root_link_pos_w = self.command_manager.ref_root_link_pos_w[:, None, :].clone()
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[:, None, :]
        
        robot_body_link_pos_w = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        robot_root_link_pos_w = self.command_manager.asset.data.root_link_pos_w[:, None, :].clone()
        robot_root_link_quat_w = self.command_manager.asset.data.root_link_quat_w[:, None, :]
        
        ref_root_link_pos_w[..., 2] = 0.0
        robot_root_link_pos_w[..., 2] = 0.0
        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_body_pos_local = quat_apply_inverse(ref_root_link_quat_w, ref_body_link_pos_w - ref_root_link_pos_w)
        robot_body_pos_local = quat_apply_inverse(robot_root_link_quat_w, robot_body_link_pos_w - robot_root_link_pos_w)

        # shape: [num_envs, num_tracking_bodies, 3]
        body_pos_error = (ref_body_pos_local - robot_body_pos_local).norm(dim=-1)
        self.error[:] = body_pos_error.max(dim=1).values
        super().update()
    
class cum_body_ori_error_local(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, body_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.body_names = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)[1]
        self.body_indices_asset = [self.command_manager.asset.body_names.index(name) for name in self.body_names]
        self.body_indices_motion = [self.command_manager.tracking_keypoint_names.index(name) for name in self.body_names]

    def update(self):
        ref_body_link_quat_w = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[:, None, :]
        
        robot_body_link_quat_w = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        robot_root_link_quat_w = self.command_manager.asset.data.root_link_quat_w[:, None, :]
        
        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w).expand_as(ref_body_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w).expand_as(robot_body_link_quat_w)

        ref_body_quat_local = quat_mul(quat_conjugate(ref_root_link_quat_w), ref_body_link_quat_w)
        robot_body_quat_local = quat_mul(quat_conjugate(robot_root_link_quat_w), robot_body_link_quat_w)

        # shape: [num_envs, num_tracking_bodies, 3]
        body_quat_diff = quat_mul(quat_conjugate(ref_body_quat_local), robot_body_quat_local)
        body_ori_error = axis_angle_from_quat(body_quat_diff).norm(dim=-1)
        self.error[:] = body_ori_error.max(dim=1).values
        super().update()
    
class cum_joint_pos_error(_cum_error_mixin, RobotTrackTermination):
    def __init__(self, joint_names: str | List[str] = ".*", **kwargs):
        super().__init__(**kwargs)
        self.joint_names = resolve_matching_names(joint_names, self.command_manager.tracking_joint_names)[1]
        self.joint_indices_asset = [self.command_manager.asset.joint_names.index(name) for name in self.joint_names]
        self.joint_indices_motion = [self.command_manager.tracking_joint_names.index(name) for name in self.joint_names]

    def update(self):
        ref_joint_pos = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        robot_joint_pos = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]

        joint_pos_error = (ref_joint_pos - robot_joint_pos).abs()
        self.error[:] = joint_pos_error.max(dim=1).values
        super().update()

RobotObjectTrackTermination = BaseTermination[RobotObjectTracking]

class cum_object_pos_error(_cum_error_mixin, RobotObjectTrackTermination):
    def update(self):
        ref_object_pos_w = self.command_manager.ref_object_pos_w
        box_pos_w = self.command_manager.object.data.root_link_pos_w
        box_pos_diff = ref_object_pos_w - box_pos_w
        self.error[:] = box_pos_diff.norm(dim=-1)
        super().update()

class cum_object_ori_error(_cum_error_mixin, RobotObjectTrackTermination):
    def update(self):
        ref_object_quat_w = self.command_manager.ref_object_quat_w
        object_quat_w = self.command_manager.object.data.root_link_quat_w
        box_quat_diff = quat_mul(quat_conjugate(object_quat_w), ref_object_quat_w)
        self.error[:] = axis_angle_from_quat(box_quat_diff).norm(dim=-1)
        super().update()

class cum_lost_contact_steps(_cum_error_mixin, RobotObjectTrackTermination):
    def __init__(self, pos_thres: float=0.05, frc_thres: float=2.0, threshold: float=1.0, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.pos_thres = pos_thres
        self.frc_thres = frc_thres
        if isinstance(frc_thres, ListConfig):
            self.frc_thres = torch.tensor(frc_thres, device=self.device)
    
    def update(self):
        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        eef_frc = self.command_manager.eef_contact_forces_b

        contact_pos = eef_pos_diff.norm(dim=-1) < self.pos_thres
        if isinstance(self.frc_thres, float):
            contact_frc = (eef_frc.norm(dim=-1) >= self.frc_thres)
        else:
            contact_frc = (eef_frc.abs() >= self.frc_thres).all(dim=-1)

        in_contact = contact_pos & contact_frc
        in_range = self.command_manager.ref_object_contact
        lost_contact = (in_range & (~in_contact)).any(dim=-1)
        self.error[:] = 2 * lost_contact.float()
        super().update()