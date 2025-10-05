from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking, RobotObjectTracking
from active_adaptation.envs.mdp.base import Reward as BaseReward

from typing import List, Dict, Tuple
from omegaconf import DictConfig, ListConfig
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat, yaw_quat

import torch


TrackReward = BaseReward[RobotTracking]

class _tracking_keypoint(TrackReward):
    def __init__(self, body_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if body_names is None:
            body_names = self.command_manager.tracking_keypoint_names
        
        self.sigma = sigma
        body_indices_motion, matched_names_motion = resolve_matching_names(body_names, self.command_manager.tracking_keypoint_names)
        body_indices_asset, matched_names_asset = resolve_matching_names(body_names, self.command_manager.asset.body_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "body names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_keypoint_names), "Some body names in motion dataset not found in tracking body names"
        
        self.body_indices_motion = []
        self.body_indices_asset = []
        self.body_names = list(sorted(matched_names))
        self.num_bodies = len(self.body_names)
        for body_name in self.body_names:
            body_idx_motion = self.command_manager.tracking_keypoint_names.index(body_name)
            body_idx_asset = self.command_manager.asset.body_names.index(body_name)

            self.body_indices_motion.append(body_idx_motion)
            self.body_indices_asset.append(body_idx_asset)

        self.tolerance = torch.zeros(len(self.body_names), device=self.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, self.body_names)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.device)
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

    def compute(self):
        raise NotImplementedError

class keypoint_pos_tracking_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_pos_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_link_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_link_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_link_quat_w
        root_quat_motion = self.command_manager.ref_root_link_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

    def debug_draw(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_link_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_link_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_link_quat_w
        root_quat_motion = self.command_manager.ref_root_link_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)
        # self.env._debug_draw.vector(
        #     root_pos_asset,
        #     body_pos_asset_relative,
        #     color=(0.0, 1.0, 0.0),
        #     size=4.0,
        # )
        # self.env._debug_draw.vector(
        #     root_pos_motion,
        #     body_pos_motion_relative,
        #     color=(1.0, 0.0, 0.0),
        #     size=4.0,
        # )
        # self.env.debug_draw.point(
        #     body_pos_asset_relative.reshape(-1, 3),
        #     color=(0.0, 1.0, 0.0, 1.0),
        #     size=20,
        # )
        # self.env.debug_draw.point(
        #     body_pos_motion_relative.reshape(-1, 3),
        #     color=(1.0, 0.0, 0.0, 1.0),
        #     size=20,
        # )

class keypoint_pos_error(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]
        diff = body_pos_motion - body_pos_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)

class keypoint_pos_error_local(_tracking_keypoint):
    def compute(self):
        body_pos_asset = self.command_manager.asset.data.body_link_pos_w[:, self.body_indices_asset]
        body_pos_motion = self.command_manager.ref_body_link_pos_w[:, self.body_indices_motion]

        root_pos_asset = self.command_manager.robot_root_link_pos_w.clone()
        root_pos_motion = self.command_manager.ref_root_link_pos_w.clone()
        root_quat_asset = self.command_manager.robot_root_link_quat_w
        root_quat_motion = self.command_manager.ref_root_link_quat_w
        
        root_pos_asset[..., 2] = 0.0
        root_pos_motion[..., 2] = 0.0
        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)
        
        root_pos_asset = root_pos_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_pos_motion = root_pos_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_pos_asset_relative = quat_apply_inverse(root_quat_asset, body_pos_asset - root_pos_asset)
        body_pos_motion_relative = quat_apply_inverse(root_quat_motion, body_pos_motion - root_pos_motion)

        diff = body_pos_motion_relative - body_pos_asset_relative
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)


class keypoint_ori_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class keypoint_ori_tracking_local_product(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_link_quat_w
        root_quat_motion = self.command_manager.ref_root_link_quat_w

        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
        body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

        diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class keypoint_ori_error(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]
        diff = quat_mul(quat_conjugate(body_ori_motion), body_ori_asset)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)
    
class keypoint_ori_error_local(_tracking_keypoint):
    def compute(self):
        body_ori_asset = self.command_manager.asset.data.body_link_quat_w[:, self.body_indices_asset]
        body_ori_motion = self.command_manager.ref_body_link_quat_w[:, self.body_indices_motion]

        root_quat_asset = self.command_manager.robot_root_link_quat_w
        root_quat_motion = self.command_manager.ref_root_link_quat_w

        root_quat_asset = yaw_quat(root_quat_asset)
        root_quat_motion = yaw_quat(root_quat_motion)

        root_quat_asset = root_quat_asset.unsqueeze(1).expand(-1, self.num_bodies, -1)
        root_quat_motion = root_quat_motion.unsqueeze(1).expand(-1, self.num_bodies, -1)

        body_ori_asset_relative = quat_mul(quat_conjugate(root_quat_asset), body_ori_asset)
        body_ori_motion_relative = quat_mul(quat_conjugate(root_quat_motion), body_ori_motion)

        diff = quat_mul(quat_conjugate(body_ori_motion_relative), body_ori_asset_relative)
        # shape: [num_envs, num_tracking_bodies, 4]
        error = torch.norm(axis_angle_from_quat(diff), dim=-1)
        error = (error - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return error.mean(dim=1).unsqueeze(1)

class keypoint_lin_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_lin_vel_asset = self.command_manager.asset.data.body_com_lin_vel_w[:, self.body_indices_asset]
        body_lin_vel_motion = self.command_manager.ref_body_com_lin_vel_w[:, self.body_indices_motion]
        diff = body_lin_vel_motion - body_lin_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
class keypoint_ang_vel_tracking_product(_tracking_keypoint):
    def compute(self):
        body_ang_vel_asset = self.command_manager.asset.data.body_com_ang_vel_w[:, self.body_indices_asset]
        body_ang_vel_motion = self.command_manager.ref_body_com_ang_vel_w[:, self.body_indices_motion]
        diff = body_ang_vel_motion - body_ang_vel_asset
        # shape: [num_envs, num_tracking_bodies, 3]
        error = (diff.norm(dim=-1) - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_bodies]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

class _tracking_joint(TrackReward):
    def __init__(self, joint_names: List[str] | str | None = None, sigma: float = 0.03, tolerance: float | Dict[str, float] = 0.0, **kwargs):
        super().__init__(**kwargs)
        if joint_names is None:
            joint_names = self.command_manager.tracking_joint_names
    
        self.sigma = sigma
        joint_indices_asset, matched_names_asset = resolve_matching_names(joint_names, self.command_manager.asset.joint_names)
        joint_indices_motion, matched_names_motion = resolve_matching_names(joint_names, self.command_manager.tracking_joint_names)

        matched_names = set(matched_names_motion) & set(matched_names_asset)
        assert set(matched_names) == set(matched_names_motion) == set(matched_names_asset), "joint names in motion dataset and robot not matched"
        assert set(matched_names) <= set(self.command_manager.tracking_joint_names), "Some joint names in motion dataset not found in tracking joint names"

        self.joint_indices_motion = []
        self.joint_indices_asset = []
        self.joint_names = list(sorted(matched_names))
        for joint_name in self.joint_names:
            joint_idx_motion = self.command_manager.tracking_joint_names.index(joint_name)
            joint_idx_asset = self.command_manager.asset.joint_names.index(joint_name)

            self.joint_indices_motion.append(joint_idx_motion)
            self.joint_indices_asset.append(joint_idx_asset)

        self.tolerance = torch.zeros(len(self.joint_names), device=self.env.device)
        if isinstance(tolerance, float):
            self.tolerance[:] = tolerance
        elif isinstance(tolerance, DictConfig):
            tolerance = dict(tolerance)
            tolerance_indices, tolerance_names, tolerance_values = resolve_matching_names_values(tolerance, matched_names_motion)
            self.tolerance[tolerance_indices] = torch.tensor(tolerance_values, device=self.env.device)
        else:
            raise ValueError(f"Invalid tolerance type: {type(tolerance)}")

class joint_pos_tracking_product(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)
    
class joint_pos_error(_tracking_joint):
    def compute(self):
        joint_pos_asset = self.command_manager.asset.data.joint_pos[:, self.joint_indices_asset]
        joint_pos_motion = self.command_manager.ref_joint_pos[:, self.joint_indices_motion]
        diff = joint_pos_motion - joint_pos_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        return error.mean(dim=1).unsqueeze(1)
    
class joint_vel_tracking_product(_tracking_joint):
    def compute(self):
        joint_vel_asset = self.command_manager.asset.data.joint_vel[:, self.joint_indices_asset]
        joint_vel_motion = self.command_manager.ref_joint_vel[:, self.joint_indices_motion]
        diff = joint_vel_motion - joint_vel_asset
        error = (diff.abs() - self.tolerance).clamp_min(0.0)
        # shape: [num_envs, num_tracking_joints]
        return torch.exp(- error.mean(dim=1) / self.sigma).unsqueeze(1)

RobotObjectTrackReward = BaseReward[RobotObjectTracking]

class object_pos_tracking(RobotObjectTrackReward):
    def __init__(self, sigma: float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
    
    def compute(self):
        ref_object_pos_w = self.command_manager.ref_object_pos_w
        object_pos_w = self.command_manager.object_pos_w
        object_pos_error = (ref_object_pos_w - object_pos_w).norm(dim=-1)
        # shape: [num_envs]
        rew = torch.exp(- object_pos_error / self.sigma).unsqueeze(1)
        return rew

class object_ori_tracking(RobotObjectTrackReward):
    def __init__(self, sigma: float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
    
    def compute(self):
        ref_object_quat_w = self.command_manager.ref_object_quat_w
        object_quat_w = self.command_manager.object_quat_w
        object_diff_quat = quat_mul(quat_conjugate(ref_object_quat_w), object_quat_w)
        object_ori_error = torch.norm(axis_angle_from_quat(object_diff_quat), dim=-1)
        # shape: [num_envs]
        rew = torch.exp(- object_ori_error / self.sigma).unsqueeze(1)
        return rew

class object_joint_pos_tracking(RobotObjectTrackReward):
    def __init__(self, sigma: float=0.25, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
    
    def compute(self):
        ref_joint_pos = self.command_manager.ref_object_joint_pos
        object_joint_pos = self.command_manager.object_joint_pos
        joint_pos_diff = ref_joint_pos - object_joint_pos
        joint_pos_error = joint_pos_diff.abs()
        # shape: [num_envs]
        rew = torch.exp(- joint_pos_error / self.sigma).unsqueeze(1)
        return rew

class eef_contact_exp(RobotObjectTrackReward):
    def __init__(
        self,
        pos_sigma: float=0.1,
        pos_tolerance: float=0.0,
        frc_sigma: float=10.0,
        frc_thres: float | Tuple[float, float, float]=2.0,
        gain: float=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.eef_pos_error = torch.zeros(self.num_envs, self.command_manager.num_eefs, device=self.device)
        self.eef_frc = torch.zeros(self.num_envs, self.command_manager.num_eefs, 3, device=self.device)

        self.pos_sigma = pos_sigma
        self.pos_tolerance = pos_tolerance

        self.frc_sigma = frc_sigma
        self.frc_thres = frc_thres
        if isinstance(frc_thres, ListConfig):
            self.frc_thres = torch.tensor(frc_thres, device=self.device)
    
    def update(self):
        self.in_range = self.command_manager.ref_object_contact

        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        eef_frc = self.command_manager.eef_contact_forces_b

        self.eef_pos_error[:] = (eef_pos_diff.norm(dim=-1) - self.pos_tolerance).clamp_min(0.0)
        self.eef_frc[:] = eef_frc

    def compute(self):
        if isinstance(self.frc_thres, float):
            contact_frc = (self.eef_frc.norm(dim=-1) - self.frc_thres).clamp_max(0.0)
        else:
            contact_frc = (self.eef_frc.abs() - self.frc_thres).clamp_max(0.0).mean(dim=-1)

        rew = torch.exp(-self.eef_pos_error / self.pos_sigma) * torch.exp(contact_frc / self.frc_sigma)
        # shape: [num_envs]
        rew = (rew * self.in_range.float() * self.gain + 1 - self.in_range.float()).mean(dim=-1)
        return rew.unsqueeze(-1)

class eef_contact_exp_max(RobotObjectTrackReward):
    def __init__(
        self,
        pos_sigma: float=0.1,
        pos_tolerance: float=0.0,
        frc_sigma: float=10.0,
        frc_thres: float | Tuple[float, float, float]=2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eef_pos_error = torch.zeros(self.num_envs, self.command_manager.num_eefs, device=self.device)
        self.eef_ori_error = torch.zeros(self.num_envs, self.command_manager.num_eefs, 3, device=self.device)
        self.eef_frc = torch.zeros(self.num_envs, self.command_manager.num_eefs, 3, device=self.device)

        self.pos_sigma = pos_sigma
        self.pos_tolerance = pos_tolerance

        self.frc_sigma = frc_sigma
        self.frc_thres = frc_thres
        if isinstance(frc_thres, ListConfig):
            self.frc_thres = torch.tensor(frc_thres, device=self.device)
    
    def update(self):
        self.in_range = self.command_manager.ref_object_contact

        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        eef_frc = self.command_manager.eef_contact_forces_b

        self.eef_pos_error[:] = (eef_pos_diff.norm(dim=-1) - self.pos_tolerance).clamp_min(0.0)
        self.eef_frc[:] = eef_frc

    def compute(self):
        if isinstance(self.frc_thres, float):
            contact_frc = (self.eef_frc.norm(dim=-1) - self.frc_thres).clamp_max(0.0)
        else:
            contact_frc = (self.eef_frc.abs() - self.frc_thres).clamp_max(0.0).mean(dim=-1)

        rew = torch.exp(-self.eef_pos_error / self.pos_sigma) * torch.exp(contact_frc / self.frc_sigma)
        # shape: [num_envs]
        return (rew.max(dim=-1).values * self.in_range.any(dim=-1).float()).unsqueeze(-1)


class eef_contact_all(RobotObjectTrackReward):
    def __init__(
        self,
        pos_thres: float=0.1,
        frc_thres: float | Tuple[float, float, float]=2.0,
        gain: float=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.eef_pos_error = torch.zeros(self.num_envs, 2, device=self.device)
        self.eef_ori_error = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.eef_frc = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self.pos_thres = pos_thres
        self.frc_thres = frc_thres
        if isinstance(frc_thres, ListConfig):
            self.frc_thres = torch.tensor(frc_thres, device=self.device)
    
    def update(self):
        self.in_range = self.command_manager.ref_object_contact

        eef_pos_diff = self.command_manager.contact_eef_pos_w - self.command_manager.contact_target_pos_w
        eef_frc = self.command_manager.eef_contact_forces_b

        self.eef_pos_error[:] = eef_pos_diff.norm(dim=-1)
        self.eef_frc[:] = eef_frc

    def compute(self):
        contact_pos = (self.eef_pos_error < self.pos_thres)
        if isinstance(self.frc_thres, float):
            contact_frc = (self.eef_frc.norm(dim=-1) >= self.frc_thres)
        else:
            contact_frc = (self.eef_frc.abs() >= self.frc_thres).all(dim=-1)

        rew = (contact_pos & contact_frc).float()
        rew = (rew * self.in_range.float() * self.gain + 1 - self.in_range.float()).mean(dim=-1)
        # shape: [num_envs]
        return rew.unsqueeze(-1)


