import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

from active_adaptation.envs.mdp.base import Termination

class crash(Termination):
    def __init__(
        self, 
        env, 
        body_names_expr: str,
        t_thres: float = 0.,
        min_time: float = 0.,
        **kwargs
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names_expr)
        self.t_thres = t_thres
        self._decay = 0.98
        self._thres = (self.t_thres / self.env.physics_dt) * 0.9
        self.count = torch.zeros(self.num_envs, len(self.body_indices), device=self.env.device)
        self.min_steps = int(min_time / self.env.step_dt)
        print(f"Terminate upon contact on {self.body_names}")
    
    def reset(self, env_ids):
        self.count[env_ids] = 0.
    
    def update(self):
        in_contact = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(dim=-1) > 1.0
        self.count.add_(in_contact.float()).mul_(self._decay)
        
    def __call__(self):
        valid = (self.env.episode_length_buf > self.min_steps)
        undesired_contact = (self.count > self._thres).any(-1)
        return (undesired_contact & valid).reshape(self.num_envs, 1)

class soft_contact(Termination):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_indices, self.body_names = self.contact_sensor.find_bodies(body_names)
    
    def update(self):
        forces = self.contact_sensor.data.net_forces_w[:, self.body_indices].norm(dim=-1, keepdim=True)
        in_contact = (forces > 1.0).sum(dim=1)
        self.env.discount.mul_(0.4 ** in_contact)

    def __call__(self):
        return torch.zeros(self.num_envs, 1, device=self.env.device, dtype=bool)
    

class fall_over(Termination):
    def __init__(
        self, 
        env, 
        xy_thres: float=0.8,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.xy_thres = xy_thres
    
    def __call__(self):
        gravity_xy: torch.Tensor = self.asset.data.projected_gravity_b[:, :2]
        fall_over = gravity_xy.norm(dim=1, keepdim=True) >= self.xy_thres
        return fall_over


class tracking_error(Termination):
    def __init__(self, env, tracking_error_threshold):
        super().__init__(env)
        self.tracking_error_threshold = tracking_error_threshold
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        return self.asset.data._tracking_error > self.tracking_error_threshold


class cum_error(Termination):
    def __init__(self, env, thres: float = 0.85, min_steps: int = 50):
        super().__init__(env)
        self.thres = torch.tensor(thres, device=self.env.device)
        self.min_steps = min_steps # tolerate the first few steps
        self.error_exceeded_count = torch.zeros(self.env.num_envs, 1, device=self.env.device, dtype=torch.int32)
        self.command_manager = self.env.command_manager
    
    def reset(self, env_ids):
        self.error_exceeded_count[env_ids] = 0

    def update(self):
        error_exceeded = (self.command_manager._cum_error > self.thres).any(-1, True)
        self.error_exceeded_count[error_exceeded] += 1
        self.error_exceeded_count[~error_exceeded] = 0
    
    def __call__(self) -> torch.Tensor:
        return (self.error_exceeded_count > self.min_steps).reshape(-1, 1)

class ee_cum_error(Termination):
    def __init__(self, env, thres: float = 1.0, min_steps: int = 50):
        super().__init__(env)
        from .commands import CommandEEPose_Cont
        self.thres = torch.as_tensor(thres, device=self.env.device)
        self.min_steps = min_steps
        self.command_manager: CommandEEPose_Cont = self.env.command_manager
    
    def __call__(self) -> torch.Tensor:
        a = (self.command_manager._cum_error > self.thres).any(-1)
        b = self.env.episode_length_buf > self.min_steps
        return (a & b).reshape(-1, 1)


class joint_acc_exceeds(Termination):
    def __init__(self, env, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
    
    def __call__(self) -> torch.Tensor:
        valid = (self.env.episode_length_buf > 2).unsqueeze(-1)
        return (
            valid & 
            (self.asset.data.joint_acc.abs() > self.thres).any(1, True)
        )

class impact_exceeds(Termination):
    def __init__(self, env, body_names: str, thres: float):
        super().__init__(env)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.body_ids = self.contact_sensor.find_bodies(body_names)[0]
    
    def __call__(self) -> torch.Tensor:
        impact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.body_ids]
        return (impact_force.norm(dim=-1).mean(1) > self.thres).any(1, True)


class impedance_pos_error(Termination):
    def __init__(self, env, thres: float = 0.3):
        super().__init__(env)
        self.thres = thres
        self.command_manger = self.env.command_manager
        self.asset: Articulation = self.env.scene["robot"]

    def __call__(self):
        error = (self.asset.data.root_link_pos_w-self.command_manger.des_pos_w)[:, :2].norm(dim=-1, keepdim=True)
        return error > self.thres