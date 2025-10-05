from active_adaptation.envs.mdp.base import Reward

import torch
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    # from isaaclab.assets.articulation import Articulation
    from mjlab.entity import Entity as Articulation
    from isaaclab.sensors import ContactSensor
    
class feet_slip(Reward):
    def __init__(
        self, env, body_names: str, weight: float, tolerance: float = 0.0, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)

        self.tolerance = tolerance

    def compute(self) -> torch.Tensor:
        in_contact = (
            self.contact_sensor.data.current_contact_time[:, self.body_ids] > 0.02
        )
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (in_contact * feet_vel).sum(dim=1, keepdim=True)
        return -slip

class feet_upright(Reward):
    def __init__(
        self, env, body_names: str, xy_sigma: float, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        
        self.body_ids_asset, _ = self.asset.find_bodies(body_names)
        self.body_ids_contact, _ = self.contact_sensor.find_bodies(body_names)

        down = torch.tensor([0.0, 0.0, -1.0], device=self.env.device)
        self.down = down.expand(self.num_envs, len(self.body_ids_asset), -1)
        self.xy_sigma = xy_sigma
        
    def compute(self):
        feet_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids_asset]
        feet_projected_down = quat_apply_inverse(feet_quat_w, self.down)
        feet_projected_down_xy = feet_projected_down[:, :, :2].norm(dim=-1)
        # shape: (num_envs, num_feet)
        rew = (torch.exp(-feet_projected_down_xy / self.xy_sigma) - 1.0)
        return rew.float().mean(dim=1, keepdim=True)

class feet_close_xy(Reward):
    def __init__(self, env, body_names: str, thres: float=0.1, weight: float=1.0, enabled: bool=True):
        super().__init__(env, weight, enabled)
        self.threshold = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        assert len(self.body_ids) == 2, "Only support two feet"

    def compute(self):
        feet_pos = self.asset.data.body_link_pos_w[:, self.body_ids]
        distance_xy = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
        penalty = (distance_xy - self.threshold).clamp_max(0.0)
        return penalty.unsqueeze(1)

class feet_stumble(Reward):
    def __init__(self, env, body_names: str | List[str], weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_contact_ids = self.contact_forces.find_bodies(body_names)[0]

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_forces.data.net_forces_w[:, self.feet_contact_ids, :2].norm(dim=2) > 0.5
        return -in_contact.float().mean(1, keepdim=True)
        

class impact_force_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.default_mass_total = (
            self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        )
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        print(f"Penalizing impact forces on {self.body_names}.")

    def compute(self) -> torch.Tensor:
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        contact_forces = self.contact_sensor.data.net_forces_w_history.norm(
            dim=-1
        ).mean(1)
        force = contact_forces[:, self.body_ids] / self.default_mass_total
        return -(force.square() * first_contact).clamp_max(10.0).sum(1, True)


class feet_air_time(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
        soft_discount: float = 1.0,
        condition_on_linvel: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.condition_on_linvel = condition_on_linvel
        self.soft_discount = soft_discount

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.reward = torch.zeros(self.num_envs, 1, device=self.env.device)

        if self.env.backend != "isaac":
            return
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        self.vis_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Feet_contact",
                markers={"feet": sim_utils.SphereCfg(
                    radius=0.06, 
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)))}
            )
        )
        self.vis_marker_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3, device=self.env.device)

    def compute(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        last_air_time = self.contact_sensor.data.last_air_time[:, self.body_ids]
        self.reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * first_contact, dim=1, keepdim=True
        )
        self.reward *= ~self.env.command_manager.is_standing_env
        violation = ((last_air_time < self.thres) & first_contact).any(dim=1)
        self.env.discount[violation] = self.soft_discount
        return self.reward
    
    def debug_draw(self):
        if self.env.backend != "isaac":
            return
        self.vis_marker_pos_w.fill_(-100)
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.articulation_body_ids]
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ]
        self.vis_marker_pos_w[first_contact] = feet_pos_w[first_contact]
        self.vis_marker.visualize(
            translations=self.vis_marker_pos_w.reshape(-1, 3),
        )


class max_feet_height(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        target_height: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.target_height = target_height

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(body_names)

        self.in_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.impact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.detach = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.has_impact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.max_height = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.impact_point = torch.zeros(
            self.num_envs, len(self.body_ids), 3, device=self.device
        )
        self.detach_point = torch.zeros(
            self.num_envs, len(self.body_ids), 3, device=self.device
        )

    def reset(self, env_ids):
        self.has_impact[env_ids] = False

    def update(self):
        contact_force = self.contact_sensor.data.net_forces_w_history[
            :, :, self.body_ids
        ]
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.asset_body_ids]
        in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact[:] = (~self.in_contact) & in_contact
        self.detach[:] = self.in_contact & (~in_contact)
        self.in_contact[:] = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
        self.max_height[:] = torch.where(
            self.detach,
            feet_pos_w[:, :, 2],
            torch.maximum(self.max_height, feet_pos_w[:, :, 2]),
        )

    def compute(self) -> torch.Tensor:
        reference_height = torch.maximum(
            self.impact_point[:, :, 2], self.detach_point[:, :, 2]
        )
        max_height = self.max_height - reference_height
        # r = (self.impact * (max_height / self.target_height).clamp_max(1.0)).sum(
        #     dim=1, keepdim=True
        # )
        # this should be penalty, otherwise encourages the feet to contact more often
        penalty = self.impact * (1 - max_height / self.target_height).clamp_min(0.0)
        r = -penalty.sum(dim=1, keepdim=True)
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        # sometimes the policy can decied is_standing, so we need to set the mean reward to 0
        # r[~is_standing] -= r[~is_standing].mean()
        r[is_standing] = 0
        return r

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.asset_body_ids]
        self.env.debug_draw.point(
            feet_pos_w[self.impact],
            color=(1.0, 0.0, 0.0, 1.0),
            size=30,
        )

class feet_contact_count(Reward):
    def __init__(
        self, env, body_names: str, weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)
