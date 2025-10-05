from active_adaptation.envs.mdp.base import Observation
import active_adaptation.utils.symmetry as sym_utils

from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse
import torch
from typing import TYPE_CHECKING, Tuple, List
if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from isaaclab.sensors import ContactSensor

def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3., 3.) * std

class root_ang_vel_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        root_ang_vel_b = self.asset.data.root_com_ang_vel_b[env_ids]
        root_ang_vel_b = root_ang_vel_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer[env_ids] = root_ang_vel_b

    def update(self):
        root_ang_vel_b = self.asset.data.root_com_ang_vel_b
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_ang_vel_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

class projected_gravity_history(Observation):
    def __init__(self, env, noise_std: float=0., history_steps: list[int]=[1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()
    
    def reset(self, env_ids):
        projected_gravity_b = self.asset.data.projected_gravity_b[env_ids]
        projected_gravity_b = projected_gravity_b.unsqueeze(1).expand(-1, self.buffer.shape[1], -1)
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer[env_ids] = self.asset.data.projected_gravity_b[env_ids].unsqueeze(1)
    
    def update(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = projected_gravity_b
    
    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)

class joint_pos_history(Observation):
    def __init__(
        self,
        env,
        joint_names: str=".*",
        history_steps: list[int]=[1], 
        noise_std: float=0.,
        set_to_zero_joint_names: str | None=None
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.)
        self.asset: Articulation = self.env.scene["robot"]
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = self.env.action_manager
        self.joint_pos_offset = action_manager.offset
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.joint_mask = torch.ones(self.num_joints, device=self.device)
        if set_to_zero_joint_names is not None:
            from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
            set_to_zero_joint_ids, _ = resolve_matching_names(set_to_zero_joint_names, self.joint_names)
            self.joint_mask[set_to_zero_joint_ids] = 0.
        self.joint_mask = self.joint_mask.unsqueeze(0).unsqueeze(0) # [1, 1, J]

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.joint_pos = torch.zeros(self.num_envs, 2, self.num_joints, device=self.device)
        self.buffer = torch.zeros(shape, device=self.device)
    
    def post_step(self, substep):
        self.joint_pos[:, substep % 2] = self.asset.data.joint_pos[:, self.joint_ids]
    
    def reset(self, env_ids):
        joint_pos = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)]
        joint_pos2 = self.asset.data.joint_pos[env_ids][:, self.joint_ids]
        assert torch.allclose(joint_pos, joint_pos2)
        self.buffer[env_ids] = self.asset.data.joint_pos[env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)].unsqueeze(1)
    
    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_pos = self.joint_pos.mean(1)
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer[:, 0] = joint_pos
    
    def compute(self):
        joint_pos = self.buffer - self.joint_pos_offset[:, self.joint_ids].unsqueeze(1)
        joint_pos = joint_pos * self.joint_mask
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)
 
class prev_actions(Observation):
    def __init__(self, env, steps: int=1, flatten: bool=True, permute: bool=False):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.permute = permute
        self.action_manager = self.env.action_manager
    
    def compute(self):
        action_buf = self.action_manager.action_buf[:, :, :self.steps].clone()
        if self.permute:
            action_buf = action_buf.permute(0, 2, 1)
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        assert self.permute
        transform = self.action_manager.symmetry_transforms()
        return transform.repeat(self.steps)


class applied_action(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.action_manager.applied_action

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms()
        return transform


class applied_torque(Observation):
    def __init__(self, env, joint_names: str=".*"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
    
    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.actuator_force
        return applied_efforts[:, self.joint_ids]
    
    def symmetry_transforms(self):
        transform = sym_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform


class last_contact(Observation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]

        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.has_contact = torch.zeros(self.num_envs, len(self.body_ids), 1, dtype=bool)
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.articulation_body_ids]
        
    def reset(self, env_ids: torch.Tensor):
        self.has_contact[env_ids] = False
    
    def update(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.body_ids].unsqueeze(-1)
        self.has_contact.logical_or_(first_contact)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[:, self.articulation_body_ids]
        self.last_contact_pos_w = torch.where(
            first_contact,
            self.body_link_pos_w,
            self.last_contact_pos_w
        )
    
    def compute(self):
        distance_xy = (self.body_link_pos_w[:, :, :2] - self.last_contact_pos_w[:, :, :2]).norm(dim=-1)
        distance_z = self.body_link_pos_w[:, :, 2] - self.last_contact_pos_w[:, :, 2]
        distance = torch.stack([distance_xy, distance_z], dim=-1)
        return (distance * self.has_contact).reshape(self.num_envs, -1)

    def debug_draw(self):
        if self.env.sim.has_gui() and self.env.backend == "isaac":
            self.env.debug_draw.vector(
                self.body_link_pos_w,
                torch.where(self.has_contact, self.last_contact_pos_w, self.body_link_pos_w) - self.body_link_pos_w
            )


class jacobians_b(Observation):
    """The jacobians relative to the root link in body frame. The shape of returned jacobian is (num_envs, num_bodies * 6 * num_joints)"""
    def __init__(self, env, body_names: str, joint_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        if self.env.fix_root_link:
            self.body_ids = self.body_ids - 1
        else:
            self.joint_ids = self.joint_ids + 6
    
    def compute(self) -> torch.Tensor:
        jacobian_all = self.asset.root_physx_view.get_jacobians() # [N, B, 6, J]
        jacobian = jacobian_all[:, self.body_ids.unsqueeze(1), :, self.joint_ids.unsqueeze(0)].permute(2, 0, 3, 1) # [N, b, j, 6]
        root_link_quat_w = self.asset.data.root_link_quat_w # [N, 4]
        # [N, b, 6, j] -> [N, b, j, 6] -> [N, b * j * 2, 3] then rotate
        jacobian_b = jacobian.permute(0, 1, 3, 2).reshape(self.num_envs, -1, 3)
        jacobian_b = quat_apply_inverse(root_link_quat_w.unsqueeze(1), jacobian_b)

        # # [N, b * j * 2, 3] -> [N, b * j, 6] -> [N, b, j, 6] -> [N, b, 6, j]
        # jacobian_b = jacobian_b.reshape(self.num_envs, len(self.body_ids), -1, 6).permute(0, 1, 3, 2)
        # arm_joint_ids, _ = self.asset.find_joints("arm_joint[1-6]")
        # breakpoint()

        return jacobian_b.reshape(self.num_envs, -1)


class random_noise_placeholder(Observation):
    def __init__(self, env, dim: int, noise_std: float=1.0):
        self.noise_std = noise_std
        super().__init__(env)
        self.dim = dim
    
    def compute(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.dim, device=self.device).clamp(-3, 3) * self.noise_std

class depth_camera(Observation):
    def __init__(
        self,
        env,
        camera_name: str,
        max_depth: float = 4.0,
        min_depth: float = 0.1,
        nan_to_num: float = 4.0,
        # Domain Randomization Parameters
        gaussian_filter_kernel_choices: List[int] = [1, 3, 5],
        gaussian_filter_sigma_range: Tuple[float, float] = (1.2, 1.2),
        noise_std: Tuple[float, float] = 0.05,  # 5cm
        episode_noise_range: Tuple[float, float] = (-0.15, 0.15), # 15cm
        delay_range: Tuple[int, int] = (0, 8), # frames
    ):
        super().__init__(env)
        from isaaclab.sensors import TiledCamera
        self.camera_name = camera_name
        self.camera: TiledCamera = self.env.scene.sensors[camera_name]

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.nan_to_num = nan_to_num

        # step noise
        self.noise_std = noise_std

        # episode DR
        self.gaussian_filter_kernel_choices = gaussian_filter_kernel_choices
        self.gaussian_filter_sigma_range = gaussian_filter_sigma_range
        self.episodic_noise_range = episode_noise_range
        self.delay_range = delay_range

        with torch.device(self.device):
            self.gaussian_filter_sigma = torch.zeros(self.num_envs, dtype=torch.float32)

            self.gaussian_filter_kernel_choices = torch.tensor(self.gaussian_filter_kernel_choices, dtype=torch.int32)
            self.gaussian_filter_kernel = torch.zeros(self.num_envs, dtype=torch.int32)

            self.episodic_noise = torch.zeros(self.num_envs, dtype=torch.float32)

            self.depth_img_buffer = torch.zeros((self.num_envs, self.delay_range[1] + 1, *self.camera.image_shape))
            self.delay = torch.zeros(self.num_envs, dtype=torch.int32)

            self.all_env_ids = torch.arange(self.num_envs, dtype=torch.int32)

        # TODO: implement DR for camera intrinsics and extrinsics
        # self.camera.set_intrinsic_matrices()
        # self.camera.set_world_poses()
    
    def reset(self, env_ids):
        # resample Dr parameters
        with torch.device(self.device):
            self.gaussian_filter_sigma[env_ids] = torch.empty(len(env_ids)).uniform_(*self.gaussian_filter_sigma_range)

            choice_of_kernel = torch.randint(low=0, high=len(self.gaussian_filter_kernel_choices), size=(len(env_ids),))
            self.gaussian_filter_kernel[env_ids] = self.gaussian_filter_kernel_choices[choice_of_kernel]

            self.episodic_noise[env_ids] = torch.empty(len(env_ids)).uniform_(*self.episodic_noise_range)
            
            self.delay[env_ids] = torch.randint(
                low=self.delay_range[0], 
                high=self.delay_range[1] + 1, 
                size=(len(env_ids),),
                dtype=torch.int32
            )
    
    def compute(self) -> torch.Tensor:
        depth_img_current = self.camera.data.output["depth"].squeeze(-1)  # [N, H, W]
        # update the depth image buffer with the latest depth image
        self.depth_img_buffer = self.depth_img_buffer.roll(1, dims=1)
        self.depth_img_buffer[:, 0] = depth_img_current
        depth_img = self.depth_img_buffer[self.all_env_ids, self.delay, :, :]
        
        # TODO: does not support different kernel sizes and sigmas for each environment, not batched
        # # gaussian filter with the specified kernel size and sigma
        # import torchvision.transforms.functional as TF
        # depth_img = TF.gaussian_blur(depth_img, kernel_size=[self.gaussian_filter_kernel, self.gaussian_filter_kernel],
        #                                 sigma=self.gaussian_filter_sigma)
        # add noise
        if self.noise_std > 0:
            depth_img += torch.randn_like(depth_img) * self.noise_std
        # add deviation
        depth_img += self.episodic_noise.unsqueeze(-1).unsqueeze(-1)

        # post-process the depth image
        depth_img.nan_to_num_(nan=self.nan_to_num, posinf=self.max_depth, neginf=self.min_depth)
        depth_img.clamp_(min=self.min_depth, max=self.max_depth)
        return depth_img.unsqueeze(1) # [N, 1, H, W]