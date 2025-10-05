from active_adaptation.envs.mdp.base import Observation

import torch
# from isaaclab.assets.articulation import Articulation
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse, matrix_from_quat, quat_mul, quat_conjugate, yaw_quat
from active_adaptation.utils.math import batchify
quat_apply_inverse = batchify(quat_apply_inverse)

# obs for amp

class _history(Observation):
    obs_shape: tuple[int, ...] = (1,)

    def __init__(self, env, history_steps: list[int]=[0]):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        shape = (self.num_envs, self.buffer_size, *self.obs_shape)
        self._buffer = torch.zeros(shape, device=self.device)
        self.obs_this_step = torch.zeros(self.num_envs, *self.obs_shape, device=self.device)
    
    def update(self):
        self._buffer = self._buffer.roll(1, 1)
        self._buffer[:, 0] = self.obs_this_step
    
    def compute(self):
        return self._buffer[:, self.history_steps].reshape(self.num_envs, -1)

class joint_pos_history_amp(_history):
    def __init__(
        self,
        joint_names: str=".*",
        **kwargs
    ):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        joint_names = self.asset.find_joints(joint_names)[1]
        self.joint_names = list(sorted(joint_names))
        self.joint_ids = [self.asset.joint_names.index(name) for name in self.joint_names]
        self.num_joints = len(self.joint_ids)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.obs_shape = (self.num_joints,)

        super().__init__(**kwargs)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def update(self):
        joint_pos = self.joint_pos.mean(1)
        self.obs_this_step[:] = joint_pos
        super().update()
    
class joint_vel_history_amp(_history):
    def __init__(
        self,
        joint_names: str=".*",
        **kwargs
    ):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        joint_names = self.asset.find_joints(joint_names)[1]
        self.joint_names = list(sorted(joint_names))
        self.joint_ids = [self.asset.joint_names.index(name) for name in self.joint_names]
        self.num_joints = len(self.joint_ids)
        self.joint_vel = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.obs_shape = (self.num_joints,)

        super().__init__(**kwargs)
    
    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def update(self):
        joint_vel = self.joint_vel.mean(1)
        self.obs_this_step[:] = joint_vel
        super().update()
    
class body_pos_b_history(_history):
    def __init__(self, body_names: str, **kwargs):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        body_names = self.asset.find_bodies(body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [self.asset.body_names.index(name) for name in self.body_names]
        self.obs_shape = (len(self.body_indices), 3)
        super().__init__(**kwargs)

    def update(self):
        root_link_pos_w = self.asset.data.root_link_pos_w.clone()
        root_link_quat_w = self.asset.data.root_link_quat_w

        root_link_pos_w_flat = root_link_pos_w.clone()
        root_link_pos_w_flat[..., 2] = 0.0
        root_link_quat_w = yaw_quat(root_link_quat_w)

        body_link_pos_w = self.asset.data.body_link_pos_w[:, self.body_indices]

        body_pos_b = quat_apply_inverse(
            root_link_quat_w.unsqueeze(1),
            body_link_pos_w - root_link_pos_w_flat.unsqueeze(1)
        )
        self.obs_this_step[:] = body_pos_b
        super().update()
        
class body_lin_vel_b_history(_history):
    def __init__(self, body_names: str, **kwargs):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        body_names = self.asset.find_bodies(body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [self.asset.body_names.index(name) for name in self.body_names]
        self.obs_shape = (len(self.body_indices), 3)
        super().__init__(**kwargs)
        
    def update(self):
        root_link_quat_w = self.asset.data.root_link_quat_w
        root_link_quat_w = yaw_quat(root_link_quat_w)

        body_link_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.body_indices]
        body_lin_vel_b = quat_apply_inverse(
            root_link_quat_w.unsqueeze(1),
            body_link_lin_vel_w
        )
        self.obs_this_step[:] = body_lin_vel_b
        super().update()


class body_ori_b_history(_history):
    def __init__(self, body_names: str, **kwargs):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        body_names = self.asset.find_bodies(body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [self.asset.body_names.index(name) for name in self.body_names]
        self.obs_shape = (len(self.body_indices), 2, 3)
        super().__init__(**kwargs)

    def update(self):
        root_link_quat_w = self.asset.data.root_link_quat_w
        root_link_quat_w_yaw = yaw_quat(root_link_quat_w)

        body_link_quat_w = self.asset.data.body_link_quat_w[:, self.body_indices]
        body_quat_b = quat_mul(
            quat_conjugate(root_link_quat_w_yaw).unsqueeze(1).expand_as(body_link_quat_w),
            body_link_quat_w
        )
        body_ori_b = matrix_from_quat(body_quat_b)
        self.obs_this_step[:] = body_ori_b[:, :, :2, :3]
        super().update()

class body_ang_vel_b_history(_history):
    def __init__(self, body_names: str, **kwargs):
        self.env = kwargs["env"]
        self.asset: Articulation = self.env.scene["robot"]
        body_names = self.asset.find_bodies(body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [self.asset.body_names.index(name) for name in self.body_names]
        self.obs_shape = (len(self.body_indices), 3)
        super().__init__(**kwargs)
        
    def update(self):
        root_link_quat_w = self.asset.data.root_link_quat_w
        root_link_quat_w = yaw_quat(root_link_quat_w)

        body_link_ang_vel_w = self.asset.data.body_com_ang_vel_w[:, self.body_indices]
        body_ang_vel_b = quat_apply_inverse(
            root_link_quat_w.unsqueeze(1),
            body_link_ang_vel_w
        )
        self.obs_this_step[:] = body_ang_vel_b
        super().update()
