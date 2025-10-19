from active_adaptation.envs.mdp.commands.hdmi.command import (
    RobotTracking,
    RobotObjectTracking,
)
from active_adaptation.envs.mdp.base import Observation as BaseObservation

import torch
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_apply_inverse,
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
    yaw_quat,
    wrap_to_pi,
)
from active_adaptation.utils.math import batchify

quat_apply_inverse = batchify(quat_apply_inverse)

RobotTrackObservation = BaseObservation[RobotTracking]


class ref_joint_pos_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_pos_future_.view(self.num_envs, -1)


class ref_joint_vel_future(RobotTrackObservation):
    def compute(self):
        return self.command_manager.ref_joint_vel_future_.view(self.num_envs, -1)


class ref_joint_pos_action(RobotTrackObservation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_joint_names
        ]

    def compute(self):
        ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
            :, self.action_indices_motion
        ]
        return ref_joint_pos


class ref_joint_pos_action_policy(RobotTrackObservation):
    def __init__(self, use_joint_ctrl: bool = False, **kwargs):
        super().__init__(**kwargs)
        action_manager = self.env.action_manager
        action_joint_names = action_manager.joint_names
        self.action_indices_motion = [
            self.command_manager.dataset.joint_names.index(joint_name)
            for joint_name in action_joint_names
        ]

        self.action_scaling = action_manager.action_scaling
        self.default_joint_pos = action_manager.default_joint_pos[
            :, action_manager.joint_ids
        ]
        self.use_joint_ctrl = use_joint_ctrl

        # Check if joint_ctrl is available in the dataset
        self.has_joint_ctrl = self.command_manager.dataset.data.joint_ctrl is not None

        if self.use_joint_ctrl and not self.has_joint_ctrl:
            raise ValueError(
                "use_joint_ctrl=True but motion data does not contain joint_ctrl"
            )

    def compute(self):
        if self.use_joint_ctrl:
            # Return reference control (torque/force) directly
            ref_joint_pos = self.command_manager.current_ref_motion.joint_ctrl[
                :, self.action_indices_motion
            ]
        else:
            # Return reference joint position as action
            ref_joint_pos = self.command_manager.current_ref_motion.joint_pos[
                :, self.action_indices_motion
            ]
        ref_joint_action = (
            ref_joint_pos - self.default_joint_pos
        ) / self.action_scaling
        return ref_joint_action


class ref_root_pos_future_b(RobotTrackObservation):
    """
    Reference root position in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_pos_future_b = torch.zeros(
            self.num_envs, num_future_steps, 3, device=self.device
        )

    def update(self):
        ref_root_pos_future_w = (
            self.command_manager.ref_root_pos_future_w
        )  # shape: [num_envs, num_future_steps, 3]
        robot_root_link_pos_w = self.command_manager.robot_root_link_pos_w[
            :, None, :
        ]  # shape: [num_envs, 1, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_pos_future_b = quat_apply_inverse(
            robot_root_link_quat_w, ref_root_pos_future_w - robot_root_link_pos_w
        )
        self.ref_root_pos_future_b = ref_root_pos_future_b

    def compute(self):
        return self.ref_root_pos_future_b.view(self.num_envs, -1)


class ref_root_ori_future_b(RobotTrackObservation):
    """
    Reference root orientation in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        num_future_steps = self.command_manager.num_future_steps
        self.ref_root_ori_future_b = torch.zeros(
            self.num_envs, num_future_steps, 2, 3, device=self.device
        )

    def update(self):
        ref_root_quat_future_w = (
            self.command_manager.ref_root_quat_future_w
        )  # shape: [num_envs, num_future_steps, 4]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_quat_future_b = quat_mul(
            quat_conjugate(robot_root_link_quat_w).expand_as(ref_root_quat_future_w),
            ref_root_quat_future_w,
        )
        ref_root_ori_future_b = matrix_from_quat(ref_root_quat_future_b)
        self.ref_root_ori_future_b = ref_root_ori_future_b[:, :, :2, :]

    def compute(self):
        return self.ref_root_ori_future_b.reshape(self.num_envs, -1)


class ref_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in motion root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_pos_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_pos_future_w = (
            self.command_manager.ref_body_pos_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_link_pos_w = self.command_manager.ref_root_link_pos_w[
            :, None, None, :
        ].clone()  # shape: [num_envs, 1, 1, 3]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]

        ref_root_link_pos_w[..., 2] = 0.0
        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(
            ref_root_link_quat_w, ref_body_pos_future_w - ref_root_link_pos_w
        )
        self.ref_body_pos_future_local = ref_body_pos_future_local

    def compute(self):
        return self.ref_body_pos_future_local.view(self.num_envs, -1)


class ref_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_body_ori_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_quat_future_w = (
            self.command_manager.ref_body_quat_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]

        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_link_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w,
        )
        self.ref_body_ori_future_local = matrix_from_quat(ref_body_quat_future_local)

    def compute(self):
        return self.ref_body_ori_future_local[:, :, :, :2, :].reshape(self.num_envs, -1)


class diff_body_pos_future_local(RobotTrackObservation):
    """
    Reference body position in each motion root frame - Robot body position in robot root frame.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_pos_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_pos_future_w = (
            self.command_manager.ref_body_pos_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_link_pos_w = self.command_manager.ref_root_link_pos_w[
            :, None, None, :
        ].clone()  # shape: [num_envs, 1, 1, 3]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]

        robot_body_link_pos_w = (
            self.command_manager.robot_body_link_pos_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_link_pos_w = self.command_manager.robot_root_link_pos_w[
            :, None, :
        ].clone()  # shape: [num_envs, 1, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_link_pos_w[..., 2] = 0.0
        robot_root_link_pos_w[..., 2] = 0.0
        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_body_pos_future_local = quat_apply_inverse(
            ref_root_link_quat_w, ref_body_pos_future_w - ref_root_link_pos_w
        )
        robot_body_pos_local = quat_apply_inverse(
            robot_root_link_quat_w, robot_body_link_pos_w - robot_root_link_pos_w
        )

        self.diff_body_pos_future_local = (
            ref_body_pos_future_local - robot_body_pos_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_pos_future_local.view(self.num_envs, -1)


class diff_body_lin_vel_future_local(RobotTrackObservation):
    """
    Reference body linear velocity in motion root frame - Robot body linear velocity in robot root frame.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_lin_vel_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_lin_vel_future_w = (
            self.command_manager.ref_body_lin_vel_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_lin_vel_w = (
            self.command_manager.robot_body_com_lin_vel_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_body_lin_vel_future_local = quat_apply_inverse(
            ref_root_link_quat_w, ref_body_lin_vel_future_w
        )
        robot_body_lin_vel_local = quat_apply_inverse(
            robot_root_link_quat_w, robot_body_link_lin_vel_w
        )

        self.diff_body_lin_vel_future_local = (
            ref_body_lin_vel_future_local - robot_body_lin_vel_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_lin_vel_future_local.view(self.num_envs, -1)


class diff_body_ori_future_local(RobotTrackObservation):
    """
    Reference body orientation in motion root frame - Robot body orientation in robot root frame.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ori_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_quat_future_w = (
            self.command_manager.ref_body_quat_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 4]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_quat_w = (
            self.command_manager.robot_body_link_quat_w
        )  # shape: [num_envs, num_tracking_bodies, 4]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_body_quat_future_local = quat_mul(
            quat_conjugate(ref_root_link_quat_w).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_w,
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_root_link_quat_w).expand_as(robot_body_link_quat_w),
            robot_body_link_quat_w,
        ).unsqueeze(1)
        diff_body_quat_future = quat_mul(
            quat_conjugate(robot_body_quat_local).expand_as(ref_body_quat_future_w),
            ref_body_quat_future_local,
        )
        self.diff_body_ori_future_local = matrix_from_quat(diff_body_quat_future)

    def compute(self):
        return self.diff_body_ori_future_local[:, :, :, :2, :].reshape(
            self.num_envs, -1
        )


class diff_body_ang_vel_future_local(RobotTrackObservation):
    """
    Reference body linear velocity in motion root frame - Robot body linear velocity in robot root frame.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_body_ang_vel_future_local = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            self.command_manager.num_tracking_bodies,
            3,
            device=self.device,
        )

    def update(self):
        ref_body_ang_vel_future_w = (
            self.command_manager.ref_body_ang_vel_future_w
        )  # shape: [num_envs, num_future_steps, num_tracking_bodies, 3]
        ref_root_link_quat_w = self.command_manager.ref_root_link_quat_w[
            :, None, None, :
        ]  # shape: [num_envs, 1, 1, 4]
        robot_body_link_ang_vel_w = (
            self.command_manager.robot_body_com_ang_vel_w
        )  # shape: [num_envs, num_tracking_bodies, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        ref_root_link_quat_w = yaw_quat(ref_root_link_quat_w)
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_body_ang_vel_future_local = quat_apply_inverse(
            ref_root_link_quat_w, ref_body_ang_vel_future_w
        )
        robot_body_ang_vel_local = quat_apply_inverse(
            robot_root_link_quat_w, robot_body_link_ang_vel_w
        )

        self.diff_body_ang_vel_future_local = (
            ref_body_ang_vel_future_local - robot_body_ang_vel_local.unsqueeze(1)
        )

    def compute(self):
        return self.diff_body_ang_vel_future_local.view(self.num_envs, -1)


class ref_motion_phase(RobotTrackObservation):
    def compute(self):
        return (self.command_manager.t / self.command_manager.motion_len).unsqueeze(1)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return yaw


RobotObjectTrackObservation = BaseObservation[RobotObjectTracking]


class ref_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        episodic_noise_std: float = 0.0,
        yaw_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std
        self.yaw_only = yaw_only
        self.ref_contact_pos_b = torch.zeros_like(
            self.command_manager.contact_target_pos_w
        )

        self.step_noise = torch.zeros_like(self.command_manager.contact_target_pos_w)
        self.episodic_noise = torch.zeros_like(
            self.command_manager.contact_target_pos_w
        )

    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = (
                torch.empty(
                    len(env_ids),
                    *self.command_manager.contact_target_pos_w.shape[1:],
                    device=self.device,
                ).uniform_(-1, 1)
                * self.episodic_noise_std
            )

    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = (
                torch.randn_like(self.command_manager.contact_target_pos_w).clamp(-3, 3)
                * self.noise_std
            )

        ref_contact_target_pos_w = (
            self.command_manager.contact_target_pos_w
        )  # shape: [num_envs, n, 3]
        robot_root_link_pos_w = self.command_manager.robot_root_link_pos_w[
            :, None, :
        ]  # shape: [num_envs, 1, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        if self.yaw_only:
            robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        ref_contact_pos_b = quat_apply_inverse(
            robot_root_link_quat_w, ref_contact_target_pos_w - robot_root_link_pos_w
        )
        if self.noise_std > 0.0:
            noise = torch.randn_like(ref_contact_pos_b).clamp(-1, 1) * self.noise_std
            ref_contact_pos_b += noise
        self.ref_contact_pos_b = (
            ref_contact_pos_b + self.episodic_noise + self.step_noise
        )

    def compute(self):
        return self.ref_contact_pos_b.view(self.num_envs, -1)


class diff_contact_pos_b(RobotObjectTrackObservation):
    """
    Reference end-effector target position in robot root frame - Robot end-effector position in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_contact_pos_b = torch.zeros_like(
            self.command_manager.contact_target_pos_w
        )

    def update(self):
        ref_contact_target_pos_w = (
            self.command_manager.contact_target_pos_w
        )  # shape: [num_envs, n, 3]
        contact_eef_pos_w = (
            self.command_manager.contact_eef_pos_w
        )  # shape: [num_envs, n, 3]
        robot_root_link_quat_w = self.command_manager.robot_root_link_quat_w[
            :, None, :
        ]  # shape: [num_envs, 1, 4]

        diff_contact_pos_w = ref_contact_target_pos_w - contact_eef_pos_w
        self.diff_contact_pos_b = quat_apply_inverse(
            robot_root_link_quat_w, diff_contact_pos_w
        )

    def compute(self):
        return self.diff_contact_pos_b.view(self.num_envs, -1)


class object_xy_b(RobotObjectTrackObservation):
    """
    Object position in robot root frame
    """

    def __init__(
        self, noise_std: float = 0.0, episodic_noise_std: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.object_xy_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        self.step_noise = torch.zeros(self.num_envs, 2, device=self.device)
        self.episodic_noise = torch.zeros(self.num_envs, 2, device=self.device)

    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = (
                torch.empty(len(env_ids), 2, device=self.device).uniform_(-1, 1)
                * self.episodic_noise_std
            )

    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = (
                torch.randn_like(self.object_xy_b).clamp(-3, 3) * self.noise_std
            )
        object_pos_w = (
            self.command_manager.object.data.root_link_pos_w
        )  # shape: [num_envs, 3]
        robot_root_link_pos_w = (
            self.command_manager.robot_root_link_pos_w
        )  # shape: [num_envs, 3]
        robot_root_link_quat_w = (
            self.command_manager.robot_root_link_quat_w
        )  # shape: [num_envs, 4]
        robot_root_link_quat_w = yaw_quat(robot_root_link_quat_w)

        self.object_xy_b = (
            quat_apply_inverse(
                robot_root_link_quat_w, object_pos_w - robot_root_link_pos_w
            )[:, :2]
            + self.episodic_noise
            + self.step_noise
        )

    def compute(self):
        return self.object_xy_b.view(self.num_envs, -1)


class object_heading_b(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame
    """

    def __init__(
        self, noise_std: float = 0.0, episodic_noise_std: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.object_yaw_b = torch.zeros(self.num_envs, 1, device=self.device)
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        self.step_noise = torch.zeros_like(self.object_yaw_b)
        self.episodic_noise = torch.zeros_like(self.object_yaw_b)

    def reset(self, env_ids):
        if self.episodic_noise_std > 0.0:
            self.episodic_noise[env_ids] = (
                torch.empty(len(env_ids), 1, device=self.device).uniform_(-1, 1)
                * self.episodic_noise_std
            )

    def update(self):
        if self.noise_std > 0.0:
            self.step_noise = (
                torch.randn_like(self.object_yaw_b).clamp(-3, 3) * self.noise_std
            )
        object_quat_w = (
            self.command_manager.object.data.root_link_quat_w
        )  # shape: [num_envs, 4]
        robot_root_link_quat_w = (
            self.command_manager.robot_root_link_quat_w
        )  # shape: [num_envs, 4]

        object_yaw_w = yaw_from_quat(object_quat_w)
        robot_root_yaw_w = yaw_from_quat(robot_root_link_quat_w)

        self.object_yaw_b = (
            wrap_to_pi(object_yaw_w - robot_root_yaw_w)[:, None]
            + self.episodic_noise
            + self.step_noise
        )

    def compute(self):
        object_heading_b = torch.cat(
            [torch.cos(self.object_yaw_b), torch.sin(self.object_yaw_b)], dim=-1
        ).view(self.num_envs, -1)
        return object_heading_b


class object_pos_b(RobotObjectTrackObservation):
    """
    Object position in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_pos_b = torch.zeros(self.num_envs, 3, device=self.device)

    def update(self):
        object_pos_w = (
            self.command_manager.object.data.root_link_pos_w
        )  # shape: [num_envs, 3]
        robot_root_link_pos_w = (
            self.command_manager.robot_root_link_pos_w
        )  # shape: [num_envs, 3]
        robot_root_link_quat_w = (
            self.command_manager.robot_root_link_quat_w
        )  # shape: [num_envs, 4]

        self.object_pos_b = quat_apply_inverse(
            robot_root_link_quat_w, object_pos_w - robot_root_link_pos_w
        )

    def compute(self):
        return self.object_pos_b.view(self.num_envs, -1)


class object_ori_b(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_ori_b = torch.zeros(self.num_envs, 3, 3, device=self.device)

    def update(self):
        object_quat_w = (
            self.command_manager.object.data.root_link_quat_w
        )  # shape: [num_envs, 4]
        robot_root_link_quat_w = (
            self.command_manager.robot_root_link_quat_w
        )  # shape: [num_envs, 4]

        object_quat_b = quat_mul(
            quat_conjugate(robot_root_link_quat_w).expand_as(object_quat_w),
            object_quat_w,
        )
        self.object_ori_b = matrix_from_quat(object_quat_b)

    def compute(self):
        return self.object_ori_b.view(self.num_envs, -1)


class object_joint_pos(RobotObjectTrackObservation):
    """
    Object joint position
    """

    def compute(self):
        return self.command_manager.object_joint_pos.unsqueeze(1)


class object_joint_vel(RobotObjectTrackObservation):
    """
    Object joint velocity
    """

    def compute(self):
        return self.command_manager.object_joint_vel.unsqueeze(1)


class object_joint_torque(RobotObjectTrackObservation):
    """
    Object joint torque
    """

    def compute(self):
        return torch.zeros(self.num_envs, 1, device=self.device)
        return self.command_manager.object.custom_torques


class object_joint_friction(RobotObjectTrackObservation):
    """Object joint friction"""

    def compute(self):
        return self.command_manager.object._custom_friction.unsqueeze(1)


class object_joint_damping(RobotObjectTrackObservation):
    """Object joint damping"""

    def compute(self):
        return self.command_manager.object._custom_damping.unsqueeze(1)


class diff_object_pos_future(RobotObjectTrackObservation):
    """
    Object position in robot root frame - Robot end-effector position in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_pos_future_b = torch.zeros(
            self.num_envs, self.command_manager.num_future_steps, 3, device=self.device
        )

    def update(self):
        ref_object_pos_future_w = (
            self.command_manager.ref_object_pos_future_w
        )  # shape: [num_envs, num_future_steps, 3]
        object_pos_w = self.command_manager.object.data.root_link_pos_w.unsqueeze(1)
        diff_object_pos_future_w = ref_object_pos_future_w - object_pos_w

        object_quat_w = self.command_manager.object.data.root_link_quat_w.unsqueeze(
            1
        )  # shape: [num_envs, 1, 4]
        self.diff_object_pos_future_b = quat_apply_inverse(
            object_quat_w, diff_object_pos_future_w
        )

    def compute(self):
        return self.diff_object_pos_future_b.view(self.num_envs, -1)


class diff_object_ori_future(RobotObjectTrackObservation):
    """
    Object orientation in robot root frame - Robot end-effector orientation in robot root frame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diff_object_ori_future_b = torch.zeros(
            self.num_envs,
            self.command_manager.num_future_steps,
            3,
            3,
            device=self.device,
        )

    def update(self):
        ref_object_quat_future_w = (
            self.command_manager.ref_object_quat_future_w
        )  # shape: [num_envs, num_future_steps, 4]
        object_quat_w = self.command_manager.object.data.root_link_quat_w.unsqueeze(
            1
        )  # shape: [num_envs, 1, 4]

        diff_object_quat_future = quat_mul(
            quat_conjugate(object_quat_w).expand_as(ref_object_quat_future_w),
            ref_object_quat_future_w,
        )
        self.diff_object_ori_future_b = matrix_from_quat(diff_object_quat_future)

    def compute(self):
        return self.diff_object_ori_future_b.view(self.num_envs, -1)


class diff_object_joint_pos_future(RobotObjectTrackObservation):
    """
    Object joint position - Robot end-effector joint position
    """

    def compute(self):
        ref_object_joint_pos_future = self.command_manager.ref_object_joint_pos_future
        object_joint_pos = self.command_manager.object_joint_pos
        diff_object_joint_pos_future = (
            ref_object_joint_pos_future - object_joint_pos.unsqueeze(1)
        )
        return diff_object_joint_pos_future


class ref_object_contact_future(RobotObjectTrackObservation):
    def compute(self):
        return self.command_manager.ref_object_contact_future.view(self.num_envs, -1)
