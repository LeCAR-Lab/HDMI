import math
import torch
import numpy as np
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple

import active_adaptation

import mjlab.third_party.isaaclab.isaaclab.utils.string as string_utils
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse


if TYPE_CHECKING:
    # from isaaclab.assets import Articulation
    from mjlab.entity import Entity as Articulation


if active_adaptation.get_backend() == "isaac":
    from isaaclab.actuators import DCMotor, ImplicitActuator

from active_adaptation.envs.mdp.base import Randomization

RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


class motor_params(Randomization):
    """
    2024.10.28
    - refactor to grouped randomization.


    Example usage in the config file:

      actuator_name: base_legs
        stiffness_range:  
          F[L,R]_hip:   [0.8, 1.2]
          F[L,R]_thigh: [0.8, 1.2]
          ...
        damping_range:
          F[L,R]_hip:   [0.8, 1.2]
          F[L,R]_thigh: [0.8, 1.2]
          ...
        scale_factor_range:
          .*_hip:   [0.8, 1.2]
          .*_thigh: [0.8, 1.2]
          .*_calf:  [0.8, 1.2]
    
    """
    def __init__(
        self, 
        env,
        actuator_name,
        stiffness_range: NestedRangeType = (1.0, 1.0),
        damping_range: NestedRangeType = (1.0, 1.0),
        scale_factor_range: NestedRangeType = (1.0, 1.0),
        armature_range: NestedRangeType = (0.0, 0.0),
        strength_range: NestedRangeType = (1.0, 1.0),
        **kwargs
    ):
        super().__init__(env)
        if len(kwargs) > 0:
            import warnings
            warnings.warn(f"Got unexpected keyword arguments: {kwargs}")
        
        self.asset: Articulation = self.env.scene["robot"]
        self.actuator_name = actuator_name
        self.stiffness_range = stiffness_range
        self.damping_range = damping_range
        self.strength_range = strength_range

        self.actuator: Union[DCMotor, ImplicitActuator] = self.asset.actuators[self.actuator_name]
        self.num_joints = len(self.actuator.joint_names)
        self.default_stiffness  = self.actuator.stiffness[0].clone()
        self.default_damping    = self.actuator.damping[0].clone()
        
        from omegaconf import ListConfig
        def parse(range: NestedRangeType, default: torch.Tensor):
            if isinstance(range, (tuple, list, ListConfig)):
                range = {".*": range}
            result = {}
            for key, value in range.items():
                ids, names = string_utils.resolve_matching_names(key, self.actuator.joint_names)
                result[key] = (ids, names, value, default[ids])
            return result

        self.stiffness_range    = parse(stiffness_range, self.default_stiffness)
        self.damping_range      = parse(damping_range, self.default_damping)
        self.scale_factor_range = parse(scale_factor_range, torch.ones(self.num_joints, device=self.device))
        self.armature_range     = parse(armature_range, torch.zeros(self.num_joints, device=self.device))
        
    def reset(self, env_ids: torch.Tensor=slice(None)):
        if not self.env.backend == "isaac":
            return
        
        scale_factor = torch.ones(len(env_ids), self.num_joints, device=self.device)
        for key, (ids, names, value, default) in self.scale_factor_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            scale_factor[:, ids] = default * r

        stiffness = self.default_stiffness.expand(len(env_ids), -1).clone()
        for key, (ids, names, value, default) in self.stiffness_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            stiffness[:, ids] = default * r
        self.actuator.stiffness[env_ids] = stiffness * scale_factor
        
        damping = self.default_damping.expand(len(env_ids), -1).clone()
        for key, (ids, names, value, default) in self.damping_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            damping[:, ids] = default * r
        self.actuator.damping[env_ids] = damping * scale_factor

        armature = torch.zeros(len(env_ids), self.num_joints, device=self.device)
        for key, (ids, names, value, default) in self.armature_range.items():
            r = (value[1] - value[0]) * torch.rand(len(env_ids), 1, device=self.device) + value[0]
            armature[:, ids] = r
        self.asset.write_joint_armature_to_sim(armature, env_ids=env_ids)

        # apply randomization
        if isinstance(self.actuator, DCMotor):
            pass
        elif isinstance(self.actuator, ImplicitActuator):
            self.asset.write_joint_stiffness_to_sim(stiffness, self.actuator.joint_indices, env_ids)
            self.asset.write_joint_damping_to_sim(damping, self.actuator.joint_indices, env_ids)

# class motor_params_implicit(Randomization):
#     def __init__(self, env, actuator_name: str, stiffness_range: NestedRangeType = (1.0, 1.0), damping_range: NestedRangeType = (1.0, 1.0), scale_factor_range: NestedRangeType = (1.0, 1.0), armature_range: NestedRangeType = (0.0, 0.0)):
#         super().__init__(env)
#         self.asset: Articulation = self.env.scene["robot"]
#         self.actuator: ImplicitActuator = self.asset.actuators[actuator_name]
#         self.joint_ids, self.joint_names = self.asset.find_joints(self.actuator.joint_names)
#         self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)

#         from omegaconf import ListConfig
#         def parse(range: NestedRangeType, default: torch.Tensor):
#             if isinstance(range, (tuple, list, ListConfig)):
#                 range = {".*": range}
#             result = {}
#             for key, value in range.items():
#                 ids, names = string_utils.resolve_matching_names(key, self.joint_names)
#                 result[key] = (ids, names, value, default[ids])
#             return result
        
#         self.stiffness_range    = parse(stiffness_range, self.actuator.stiffness[0])
#         self.damping_range      = parse(damping_range, self.actuator.damping[0])
#         self.scale_factor_range = parse(scale_factor_range, torch.ones(len(self.joint_ids), device=self.device))
#         self.armature_range     = parse(armature_range, torch.zeros(len(self.joint_ids), device=self.device))

class motor_params_implicit(Randomization):
    def __init__(self, env, stiffness_range=None, damping_range=None, armature_range=None, friction_range=None):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.stiffness_range = dict(stiffness_range) if stiffness_range is not None else None
        self.damping_range = dict(damping_range) if damping_range is not None else None
        self.armature_range = dict(armature_range) if armature_range is not None else None
        self.friction_range = dict(friction_range) if friction_range is not None else None

        if self.stiffness_range is not None:
            ids, _, value = string_utils.resolve_matching_names_values(self.stiffness_range, self.asset.joint_names)
            self.stiffness_id = torch.tensor(ids, device=self.device)
            self.stiffness_default = self.asset.data.joint_stiffness[0, self.stiffness_id]
            low, high = (torch.tensor(value, device=self.device) * self.stiffness_default.unsqueeze(1)).unbind(1)
            self.stiffness_low = low
            self.stiffness_scale = high - low

        if self.damping_range is not None:
            ids, _, value = string_utils.resolve_matching_names_values(self.damping_range, self.asset.joint_names)
            self.damping_id = torch.tensor(ids, device=self.device)
            self.damping_default = self.asset.data.joint_damping[0, self.damping_id]
            low, high = (torch.tensor(value, device=self.device) * self.damping_default.unsqueeze(1)).unbind(1)
            self.damping_low = low
            self.damping_scale = high - low

        if self.armature_range is not None:
            ids, _, value = string_utils.resolve_matching_names_values(self.armature_range, self.asset.joint_names)
            self.armature_id = torch.tensor(ids, device=self.device)
            low, high = torch.tensor(value, device=self.device).unbind(1)
            self.armature_low = low
            self.armature_scale = high - low
        
        if self.friction_range is not None:
            ids, _, value = string_utils.resolve_matching_names_values(self.friction_range, self.asset.joint_names)
            self.friction_id = torch.tensor(ids, device=self.device)
            low, high = torch.tensor(value, device=self.device).unbind(1)
            self.friction_low = low
            self.friction_scale = high - low
    
    def reset(self, env_ids):
        if self.stiffness_range is not None:
            stiffness = torch.rand(len(env_ids), len(self.stiffness_id), device=self.device) * self.stiffness_scale + self.stiffness_low
            self.asset.write_joint_stiffness_to_sim(stiffness, self.stiffness_id, env_ids)

        if self.damping_range is not None:
            damping = torch.rand(len(env_ids), len(self.damping_id), device=self.device) * self.damping_scale + self.damping_low
            self.asset.write_joint_damping_to_sim(damping, self.damping_id, env_ids)

        if self.armature_range is not None:
            armature = torch.rand(len(env_ids), len(self.armature_id), device=self.device) * self.armature_scale + self.armature_low
            self.asset.write_joint_armature_to_sim(armature, self.armature_id, env_ids)

        if self.friction_range is not None:
            friction = torch.rand(len(env_ids), len(self.friction_id), device=self.device) * self.friction_scale + self.friction_low
            self.asset.write_joint_friction_coefficient_to_sim(friction, self.friction_id, env_ids)


class random_motor_failure(Randomization):
    def __init__(
        self,
        env,
        actuator_name: str,
        joint_names: str,
        failure_prob: float = 0.2,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.motors: DCMotor = self.asset.actuators[actuator_name]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names, self.motors.joint_names)
        self.joint_ids = torch.as_tensor(self.joint_ids, device=self.device)
        self.failure_prob = failure_prob
        assert not hasattr(self.motors, "motor_failure")
        self.motor_failure = self.motors.motor_failure = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)
        logging.info(f"Randomly disable one joint from {self.joint_names} with prob. {self.failure_prob}.")
        self.failure_prob = failure_prob

        # hard-coded
        self._body_ids = self.asset.find_bodies(".*calf.*")[0]
        
    def reset(self, env_ids: torch.Tensor):
        self.motor_failure[env_ids] = -1.0
        with torch.device(self.device):
            env_ids = env_ids[torch.rand(len(env_ids)) < self.failure_prob]
            i = torch.randint(0, len(self.joint_ids), env_ids.shape)
            joint_id = self.joint_ids[i]
        self.motors.stiffness[env_ids, joint_id] = 0.02
        self.motors.damping[env_ids, joint_id] = 0.02
        self.motor_failure[env_ids, i] = 1.0

    def debug_draw(self):
        x = self.asset.data.body_link_pos_w[:, self._body_ids]
        x = x[self.motor_failure > 0.]
        self.env.debug_draw.point(x, color=(0.1, 1.0, 0.1, 0.8), size=20)


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range = (0.6, 1.0),
        dynamic_friction_range = (0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool = False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError("No bodies matched the provided names for material perturbation.")

        self.homogeneous = homogeneous
        self.static_friction_range = tuple(static_friction_range)
        self.dynamic_friction_range = tuple(dynamic_friction_range)
        self.restitution_range = tuple(restitution_range)

        # Determine geoms that belong to the selected bodies.
        local_body_ids = torch.as_tensor(self.body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[local_body_ids]
        selected_body_set = set(self.global_body_ids.cpu().tolist())

        geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
        geom_names = self.asset.geom_names
        selected_geom_local: list[int] = []
        selected_geom_global: list[int] = []
        selected_geom_names: list[str] = []

        cpu_model = self.env.sim.mj_model
        for local_idx, global_idx in enumerate(geom_global_ids):
            body_id = int(cpu_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                selected_geom_local.append(local_idx)
                selected_geom_global.append(global_idx)
                selected_geom_names.append(geom_names[local_idx])

        if not selected_geom_global:
            raise ValueError(
                "No geoms found for the specified bodies when configuring material perturbation."
            )

        self.geom_local_ids = torch.as_tensor(selected_geom_local, device=self.device, dtype=torch.long)
        self.geom_global_ids = torch.as_tensor(selected_geom_global, device=self.device, dtype=torch.long)
        self.geom_names = selected_geom_names
        self._geom_global_ids_cpu = torch.as_tensor(selected_geom_global, device="cpu", dtype=torch.long)

        model = self.env.sim.model
        self._default_friction = model.geom_friction[:, self.geom_global_ids].clone()
        if hasattr(model, "geom_solref"):
            self._default_solref = model.geom_solref[:, self.geom_global_ids].clone()
        else:
            self._default_solref = None

    def _sample_range(self, range_tuple: Tuple[float, float], shape: tuple[int, int]) -> torch.Tensor:
        low, high = range_tuple
        if math.isclose(low, high):
            return torch.full(shape, low, device=self.device, dtype=torch.float32)
        return torch.rand(shape, device=self.device) * (high - low) + low

    def startup(self):
        logging.info(f"Randomize body materials of {self.geom_names} upon startup.")

        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        sample_shape = (self.num_envs, sample_cols)

        static_frictions = self._sample_range(self.static_friction_range, sample_shape)
        dynamic_frictions = self._sample_range(self.dynamic_friction_range, sample_shape)
        if sample_cols == 1:
            static_frictions = static_frictions.expand(-1, num_geoms)
            dynamic_frictions = dynamic_frictions.expand(-1, num_geoms)

        model = self.env.sim.model
        geom_friction = model.geom_friction
        geom_friction[:, self.geom_global_ids, 0] = dynamic_frictions
        # TODO: mujoco do not differentiate static and dynamic friction?
        # if geom_friction.shape[-1] > 1:
        #     geom_friction[:, self.geom_global_ids, 1] = dynamic_frictions
        # if geom_friction.shape[-1] > 2:
        #     geom_friction[:, self.geom_global_ids, 2] = dynamic_frictions

        # TODO: this is not restitution
        # if self._default_solref is not None:
        #     rest_vals = self._sample_range(self.restitution_range, sample_shape)
        #     if sample_cols == 1:
        #         rest_vals = rest_vals.expand(-1, num_geoms)
        #     geom_solref = model.geom_solref
        #     solref = self._default_solref.clone()
        #     # Map restitution to the second solref parameter (damping ratio-like term).
        #     solref[..., 1] = rest_vals
        #     geom_solref[:, self.geom_global_ids] = solref

        # Sync CPU model for viewer consistency using env 0 parameters.
        cpu_model = self.env.sim.mj_model
        cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 0] = (
            geom_friction[0, self.geom_global_ids, 0].to(device="cpu").numpy()
        )
        if geom_friction.shape[-1] > 1:
            cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 1] = (
                geom_friction[0, self.geom_global_ids, 1].to(device="cpu").numpy()
            )
        if geom_friction.shape[-1] > 2:
            cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 2] = (
                geom_friction[0, self.geom_global_ids, 2].to(device="cpu").numpy()
            )

        if self._default_solref is not None:
            cpu_model.geom_solref[self._geom_global_ids_cpu.numpy()] = (
                model.geom_solref[0, self.geom_global_ids].to(device="cpu").numpy()
            )

class perturb_body_mass(Randomization):
    def __init__(
        self, env, **perturb_ranges: Tuple[float, float]
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if not perturb_ranges:
            raise ValueError("perturb_body_mass requires at least one body range entry.")

        body_ids, body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(body_ids) == 0:
            raise ValueError("No bodies matched the provided patterns for mass perturbation.")

        device = self.device
        self.body_names = body_names
        self.local_body_ids = torch.as_tensor(body_ids, device=device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self.mass_ranges = torch.as_tensor(values, device=device, dtype=torch.float32)

        # Cache default parameters to re-apply scaling from a clean baseline.
        model = self.env.sim.model
        self._default_mass = model.body_mass[:, self.global_body_ids].clone()
        self._default_inertia = model.body_inertia[:, self.global_body_ids].clone()

        # Also cache CPU model indices for synchronization when needed.
        self._global_body_ids_cpu = self.global_body_ids.to(device="cpu", dtype=torch.long)

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")

        low = self.mass_ranges[:, 0]
        high = self.mass_ranges[:, 1]
        rand = torch.rand(self.num_envs, self.local_body_ids.numel(), device=self.device)
        scale = low + (high - low) * rand

        model = self.env.sim.model
        new_mass = self._default_mass * scale
        new_inertia = self._default_inertia * scale.unsqueeze(-1)

        model.body_mass[:, self.global_body_ids] = new_mass
        model.body_inertia[:, self.global_body_ids] = new_inertia

        # Keep CPU model (used for passive viewer) in sync with env 0 values.
        cpu_model = self.env.sim.mj_model
        cpu_model.body_mass[self._global_body_ids_cpu.numpy()] = (
            model.body_mass[0, self.global_body_ids].to(device="cpu").numpy()
        )
        cpu_model.body_inertia[self._global_body_ids_cpu.numpy()] = (
            model.body_inertia[0, self.global_body_ids].to(device="cpu").numpy()
        )

class perturb_body_com(Randomization):
    def __init__(self, env, body_names, com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError("No bodies matched the provided names for COM perturbation.")

        self.com_range = tuple(com_range)
        self.local_body_ids = torch.as_tensor(self.body_ids, device=self.device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self._global_body_ids_cpu = self.global_body_ids.to(device="cpu", dtype=torch.long)

        model = self.env.sim.model
        self._default_body_ipos = model.body_ipos[:, self.global_body_ids].clone()

    def startup(self):
        logging.info(f"Randomize COM of bodies {self.body_names} upon startup.")

        num_bodies = self.global_body_ids.numel()
        low, high = self.com_range
        offsets = torch.rand(self.num_envs, num_bodies, 3, device=self.device)
        offsets = low + (high - low) * offsets

        model = self.env.sim.model
        new_ipos = self._default_body_ipos + offsets
        model.body_ipos[:, self.global_body_ids] = new_ipos

        cpu_model = self.env.sim.mj_model
        cpu_model.body_ipos[self._global_body_ids_cpu.numpy()] = (
            model.body_ipos[0, self.global_body_ids].to(device="cpu").numpy()
        )

class JointFriction(Randomization):
    def __init__(
        self,
        env,
        friction_range=(0.01, 0.1),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.friction_range = friction_range

    def startup(self):
        logging.info("Randomize joint frictions upon starup.")
        frictions = torch.zeros(self.env.num_envs, 1)
        frictions.uniform_(*self.friction_range)
        self.asset.root_physx_view.set_dof_friction_coefficients(
            frictions.expand(-1, self.asset.num_joints), 
            indices=self.asset._ALL_INDICES.cpu()
        )


class reset_joint_states_uniform(Randomization):
    def __init__(self, env, pos_ranges: Dict[str, tuple], rel: bool=False):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.rel = rel

        self.joint_ids, self.joint_names, self.pos_ranges = string_utils.resolve_matching_names_values(
            dict(pos_ranges), self.asset.joint_names
        )
        self.pos_ranges = torch.as_tensor(self.pos_ranges, device=self.device).unbind(-1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
        self.joint_limits = self.asset.data.joint_limits[0, self.joint_ids].unbind(-1)

    def reset(self, env_ids: torch.Tensor):
        shape = (len(env_ids), len(self.joint_ids))
        init_pos = sample_uniform(shape, *self.pos_ranges, self.device)
        if self.rel:
            init_pos += self.default_joint_pos[env_ids]
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos.clamp(*self.joint_limits), 
            init_vel, self.joint_ids, env_ids #.unsqueeze(1)
        )


class reset_joint_states_scale(Randomization):
    def __init__(self, env, pos_scales: Dict[str, tuple]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        
        self.joint_ids = []
        self.pos_scales = []
        for joint_name, (low, high) in pos_scales.items():
            joint_ids, joint_names = self.asset.find_joints(joint_name)
            self.joint_ids.extend(joint_ids)
            self.pos_scales.append(torch.tensor([low, high], device=self.env.device).expand(len(joint_ids), 2))
            print(f"Reset {joint_names} to scales of U({low}, {high})")
        self.pos_scales = torch.cat(self.pos_scales, 0).unbind(1)
        self.default_joint_pos = self.asset.data.default_joint_pos[:, self.joint_ids]
        self.default_joint_vel = self.asset.data.default_joint_vel[:, self.joint_ids]
    
    def reset(self, env_ids: torch.Tensor):
        init_pos = random_scale(
            self.default_joint_pos[env_ids], 
            *self.pos_scales, 
            self.env.device
        )[0]
        init_vel = self.default_joint_vel[env_ids]
        self.asset.write_joint_state_to_sim(
            init_pos, init_vel, self.joint_ids, env_ids #.unsqueeze(1)
        )


class push(Randomization):
    def __init__(self, env, body_names, force_range = (0.2, 0.9), min_interval=100, decay: float=0.9):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81
        self.force_range = force_range
        self.min_interval = min_interval
        self.decay = decay
        
        with torch.device(self.env.device):
            self.last_push = torch.zeros(self.env.num_envs, len(self.body_indices), 1)
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        self.last_push[env_ids] = 0.

    def step(self, substep):
        if substep == 0:
            t = self.env.episode_length_buf.view(self.env.num_envs, 1, 1)
            i = torch.rand(self.env.num_envs, len(self.body_indices), 1, device=self.env.device) < 0.02
            i = i & ((t - self.last_push) > self.min_interval)
            self.last_push = torch.where(i, t, self.last_push)

            push_forces = torch.zeros_like(self.forces)
            push_forces[:, :, 0].uniform_(*self.force_range)
            push_forces[:, :, 1].uniform_(*self.force_range)
            self.forces = torch.where(i, push_forces * self.default_mass_total, self.forces * self.decay)
        # self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)
        self.asset.write_external_wrench_to_sim(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total,
            color=(1., 0.8, .4, 1.)
        )

class drag(Randomization):
    def __init__(self, env, body_names, drag_range=(0.0, 0.1)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_indices, self.body_names = self.asset.find_bodies(body_names)
        self.num_bodies = len(self.body_indices)
        self.drag_coeffs = sample_uniform((self.num_envs, self.num_bodies, 1), *drag_range, self.device).expand(self.num_envs, self.num_bodies, 3)
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum() * 9.81

        with torch.device(self.env.device):
            self.forces = torch.zeros(self.env.num_envs, len(self.body_indices), 3)
            self.torques = torch.zeros(self.env.num_envs, len(self.body_indices), 3)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.

    def step(self, substep):
        lin_vel = self.asset.data.body_com_lin_vel_w[:, self.body_indices]
        drag_forces = - lin_vel * self.drag_coeffs
        self.forces = drag_forces * self.default_mass_total
        self.asset.set_external_force_and_torque(self.forces, self.torques, body_ids=self.body_indices)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_indices],
            self.forces / self.default_mass_total * 100,
            color=(0.6, 0.8, 0.6, 1.)
        )

class stumble(Randomization):
    def __init__(
        self, 
        env,
        body_names: str,
        stumble_height: float=0.05,
        friction_range=(0.0, 0.2),
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.num_feet = len(self.body_ids)

        self.body_ids = torch.as_tensor(self.body_ids, device=self.device)
        self.stumble_height = stumble_height
        self.friction_range = friction_range
        self.friction_coef = torch.zeros(self.num_envs, 1, 1, device=self.device)
    
    def startup(self):
        self.feet_height: torch.Tensor = self.asset.data.feet_height

    def reset(self, env_ids: torch.Tensor):
        friction = torch.empty(len(env_ids), 1, 1, device=self.device)
        friction.uniform_(*self.friction_range)
        self.friction_coef[env_ids] = friction

    def step(self, substep):
        # feet_height = self.asset.data.feet_height_map.mean(-1).reshape(-1)
        feet_lin_vel_w = self.asset.data.body_com_lin_vel_w[:, self.body_ids]
        feet_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids]
        stumble_prob = ((self.stumble_height - self.feet_height) / self.stumble_height).clamp(0., 1.)
        self.forces_w = - self.friction_coef * feet_lin_vel_w / self.env.physics_dt
        self.forces_w[..., 2] = 0.
        friction_forces = torch.where(
            (torch.rand_like(self.feet_height) < stumble_prob).unsqueeze(-1),
            quat_apply_inverse(feet_quat_w, self.forces_w),
            torch.zeros(self.num_envs, self.num_feet, 3, device=self.env.device)
        )
        forces_b = self.asset._external_force_b.clone()
        torques_b = self.asset._external_torque_b.clone()
        forces_b[:, self.body_ids] += friction_forces
        self.asset.set_external_force_and_torque(forces_b, torques_b)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_ids],
            self.forces_w * self.env.physics_dt,
            color=(1., 0.6, 0., 1.)
        )


class random_joint_friction(Randomization):
    def __init__(self, env, actuator_name: str, friction_range):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joints = self.asset.actuators[actuator_name]
        self.friction_range = friction_range
    
    def startup(self):
        print(f"Randomize joint friction of joints {self.joints.joint_names}, {self.joints.joint_indices}")
        friction = torch.zeros_like(self.joints.friction)
        friction.uniform_(*self.friction_range)
        self.asset.write_joint_friction_to_sim(friction, joint_ids=self.joints.joint_indices)


class pull(Randomization):
    def __init__(
        self, 
        env,
        drag_prob: float = 0.2,
        drag_range=(0.0, 0.2)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.drag_prob = drag_prob
        self.drag_range = drag_range
        self.default_mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.device) * 9.81
        
        with torch.device(self.device):
            self.forces = torch.zeros(self.num_envs, 3)
            self.axis = torch.zeros(self.num_envs, 3)
            self.apply_drag = torch.zeros(self.num_envs, 1, dtype=bool)
            self.drag_magnitude = torch.zeros(self.num_envs, 1)

    def reset(self, env_ids: torch.Tensor):
        self.forces[env_ids] = 0.
        
        # pull direction
        a = torch.rand(len(env_ids), device=self.device) * torch.pi * 2
        axis = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
        self.axis[env_ids] = axis

        drag_magnitude = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.drag_range)
        self.drag_magnitude[env_ids] = drag_magnitude * self.default_mass_total
        self.apply_drag[env_ids] = (torch.rand(len(env_ids), 1, device=self.device) < self.drag_prob)
    
    def update(self):
        pass

    def step(self, substep):
        force =  self.axis * self.drag_magnitude
        self.forces[:] = torch.where(self.apply_drag, force, torch.zeros_like(self.forces))
        self.asset.set_external_force_and_torque(
            quat_rotate_inverse(self.asset.data.root_link_quat_w, self.forces).unsqueeze(1), 
            torch.zeros_like(force).unsqueeze(1), [0])

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w, 
            self.forces / self.default_mass_total, 
            color=(0.6, 0.8, 0.6, 1.)
        )


class external_force(Randomization):
    def __init__(self, env):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]

        with torch.device(self.device):
            self.large_force_mode = torch.zeros(self.num_envs, 1, dtype=bool)
            self.spring_force_max_range = (80., 120.)
            self.spring_force_max = torch.zeros(self.num_envs, 1)
            self.spring_force = torch.zeros(self.num_envs, 3)
            self.spring_force_duration = torch.zeros(self.num_envs, 1)
            self.spring_force_time = torch.zeros(self.num_envs, 1)
            self.spring_force_setpoint = torch.zeros(self.num_envs, 1)
            self.spring_force_kp = torch.zeros(self.num_envs, 1)
            self.spring_force_kd = torch.zeros(self.num_envs, 1)
            self.spring_end_mass = torch.zeros(self.num_envs, 1)
            self.spring_end_vel = torch.zeros(self.num_envs, 1)
    
    def reset(self, env_ids):
        self.large_force_mode[env_ids] = torch.rand(len(env_ids), 1, device=self.device) < 0.5
        self.spring_force_duration[env_ids] = -1.
        self.spring_end_mass[env_ids] = 250.

    def step(self, substep):
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(
            self.asset.data.root_link_quat_w, 
            self.spring_force
        )
        self.asset.has_external_wrench = True
    
    def update(self):
        expire = self.spring_force_time > self.spring_force_duration - 1e-4
        sample = (torch.rand(self.num_envs, 1, device=self.device) < 0.1) & expire & self.large_force_mode
        scalar = torch.zeros(self.num_envs, 1, device=self.device)
        self.spring_end_mass = torch.where(
            sample,
            torch.randint(1, 4, (self.num_envs, 1), device=self.device) * 250.,
            self.spring_end_mass
        )
        self.spring_end_vel = torch.where(
            sample,
            0.,
            self.spring_end_vel.mul_(0.98) - (self.spring_force / self.spring_end_mass) * self.env.step_dt
        )
        self.spring_force_setpoint = torch.where(
            sample,
            self.asset.data.root_link_pos_w + torch.tensor([-0.5, 0., 0.], device=self.device),
            self.spring_force_setpoint + self.spring_end_vel * self.env.step_dt
        )
        self.spring_force_kp = torch.where(
            sample,
            scalar.uniform_(80., 120.),
            self.spring_force_kp,
        )
        self.spring_force_kd = torch.where(
            sample,
            scalar.uniform_(10., 20.),
            self.spring_force_kd,
        )
        self.spring_force_duration = torch.where(
            sample,
            scalar.uniform_(2., 4.),
            self.spring_force_duration
        )
        self.spring_force_time = torch.where(
            sample,
            0.,
            self.spring_force_time + self.env.step_dt
        )
        self.spring_force_max = torch.where(
            sample,
            scalar.uniform_(*self.spring_force_max_range),
            self.spring_force_max
        )
        self.spring_force = (
            self.spring_force_kp * (self.spring_force_setpoint - self.asset.data.root_link_pos_w)
            + self.spring_force_kd * (0. - self.asset.data.root_com_lin_vel_w)    
        ) * (self.spring_force_time < self.spring_force_duration)
        self.spring_force = clamp_norm(self.spring_force, max=self.spring_force_max)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w,
            self.spring_force / (20 * 9.81),
            color=(1.0, 0.5, 0.0, 1.0),
            size=3.0,
        )


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, self.offset_range = string_utils.resolve_matching_names_values(dict(offset_range), self.asset.joint_names)
        
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.offset_range = torch.tensor(self.offset_range, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)

        self.action_manager = self.env.action_manager

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(self.offset_range[env_ids, :, 0], self.offset_range[env_ids, :, 1])
        self.action_manager.offset[env_ids.unsqueeze(1), self.joint_ids] = offset


class random_pull(Randomization):
    def __init__(self, env, force_xy_range, force_z_range, prob=0.01, duration = (0.5, 0.5)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.force_xy_range = force_xy_range
        self.force_z_range = force_z_range
        self.prob = prob
        self.duration = (duration, duration) if isinstance(duration, (int, float)) else duration
        self.mass_total = self.asset.root_physx_view.get_masses()[0].sum().to(self.device)

        self.force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.offset_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.time_remaining = torch.zeros(self.num_envs, 1, device=self.device)

    def step(self, substep):
        force_w = self.force_w * (self.time_remaining > 0)
        self.asset._external_force_b[:, 0] += quat_rotate_inverse(self.asset.data.root_link_quat_w, force_w)
        self.asset._external_torque_b[:, 0] += self.offset_b.cross(force_w, dim=-1)
        self.asset.has_external_wrench = True
    
    def update(self):
        sample_force = (torch.rand(self.num_envs, device=self.device) < self.prob).nonzero().squeeze(-1)
        if len(sample_force) > 0:
            force_w = torch.zeros(len(sample_force), 3, device=self.device)
            force_w[:, 0].uniform_(*self.force_xy_range)
            force_w[:, 1].uniform_(*self.force_xy_range)
            force_w[:, 2].uniform_(*self.force_z_range) 
            offset_b = torch.zeros(len(sample_force), 3, device=self.device)
            offset_b[:, 0].uniform_(-0.25, 0.25)
            offset_b[:, 1].uniform_(-0.15, 0.15)
            offset_b[:, 2].uniform_(-0.15, 0.15)
            self.offset_b[sample_force] = offset_b
            self.force_w[sample_force] = force_w
            duration = torch.zeros(len(sample_force), 1, device=self.device)
            duration.uniform_(*self.duration)
            self.time_remaining[sample_force] = duration / self.env.step_dt
        self.time_remaining -= 1.

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.root_link_pos_w, 
            (self.force_w / self.mass_total) * (self.time_remaining > 0), 
            color=(0.6, 0.8, 0.6, 1.)
        )


class spring_grf(Randomization):
    def __init__(self, env, feet_names: str = ".*_foot", thres_range = (0.1, 0.2), kp_range = (200, 300)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.thres_range = thres_range
        self.kp_range = kp_range

        self.feet_ids = self.asset.find_bodies(feet_names)[0]
        self.kp = torch.zeros(self.num_envs, 4, device=self.device)
        self.thres = torch.zeros(self.num_envs, 4, device=self.device)
        self.forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.flag = torch.zeros(self.num_envs, 4, dtype=bool, device=self.device)
        self.axis = torch.zeros(self.num_envs, 4, 3, device=self.device)

    def update(self):
        resample = (self.env.episode_length_buf % 100 == 0).unsqueeze(1) # [num_envs, 1]
        self.flag = torch.where(resample, torch.rand(self.flag.shape, device=self.device) < 0.2, self.flag)
        self.kp = torch.where(resample, uniform_like(self.kp, *self.kp_range), self.kp)
        self.thres = torch.where(resample, uniform_like(self.thres, *self.thres_range), self.thres)
        axis = torch.zeros(self.num_envs, 4, 3, device=self.device)
        axis[:, :, 1].uniform_(-0.3, 0.3)
        axis[:, :, 0].uniform_(-0.3, 0.3)
        axis[:, :, 2] = 1.
        axis = axis / axis.norm(dim=-1, keepdim=True)
        self.axis = torch.where(resample.unsqueeze(-1), axis, self.axis)

    def step(self, substep):
        feet_height = self.asset.data.feet_height
        feet_quat = self.asset.data.body_link_quat_w[:, self.feet_ids]
        feet_lin_vel = self.asset.data.body_com_lin_vel_w[:, self.feet_ids]
        forces = (
            self.kp * (self.thres - feet_height) + 
            5. * (0. - feet_lin_vel[:, :, 2])
        ) * self.flag
        self.forces = forces.unsqueeze(-1) * self.axis 
        self.asset._external_force_b[:, self.feet_ids] += quat_rotate_inverse(feet_quat, self.forces)
        self.asset.has_external_wrench = True

    def debug_draw(self):
        feet_pos = self.asset.data.body_link_pos_w[:, self.feet_ids]
        self.env.debug_draw.vector(feet_pos, self.forces / 9.81, color=(0.8, 0.6, 0.6, 1.))


from active_adaptation.envs.mdp.utils.forces import ImpulseForce, ConstantForce
class impulse(Randomization):
    def __init__(
        self, env,
        body_names: str = "pelvis",
        impulse_scale: Tuple[float, float, float] = (100., 100., 20.),
        duration_range: Tuple[float, float] = (0.40, 0.60),
        impulse_prob: float = 0.005,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.impulse_scale = impulse_scale
        self.duration_range = duration_range
        self.impulse_prob = impulse_prob
        self.impulse_force = self.__sample_impulse(size=self.num_envs)

        # random sample a body id
        body_ids = self.asset.find_bodies(body_names)[0]
        assert len(body_ids) == 1, "Only one body is supported"
        self.body_id = body_ids[0]
        # self.body_ids = torch.tensor(body_ids, device=self.device)
        # body_id = torch.randint(0, len(self.body_ids), (self.num_envs,), device=self.device)
        # self.body_id = self.body_ids[body_id] # shape: [num_envs]

    def __sample_impulse(self, size: int) -> ImpulseForce:
        return ImpulseForce.sample(size, self.device, self.impulse_scale, self.duration_range)
        
    def step(self, substep):
        external_wrench = self.asset.data.body_external_wrench
        forces_w, torques_w = external_wrench[..., 0:3], external_wrench[..., 3:6]
        impulse_force = self.impulse_force.get_force()
        forces_w[:, self.body_id] += impulse_force
        self.asset.write_external_wrench_to_sim(forces_w, torques_w)

    def update(self):
        expire = self.impulse_force.time > self.impulse_force.duration
        r = (torch.rand(self.num_envs, 1, device=self.device) < self.impulse_prob)
        sample = r & expire

        impulse_force = self.__sample_impulse(size=self.num_envs)

        self.impulse_force.time.add_(self.env.step_dt)
        self.impulse_force: ImpulseForce = impulse_force.where(sample, self.impulse_force)

    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[:, self.body_id],
            self.impulse_force.get_force() /  9.81,
            color=(1.0, 0.6, 0.0, 1.0),
            size=3.0,
        )


class constant_force(Randomization):
    def __init__(self, env, force_range, offset_range, body_names = None, duration_range = (1.0, 4.0)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is None:
            self.all_body_ids = torch.tensor([0], device=self.device)
        else:
            self.all_body_ids = torch.tensor(self.asset.find_bodies(body_names)[0], device=self.device)
        
        self.force = ConstantForce.sample(self.num_envs, device=self.device)
        self.force.duration.zero_()
        self.body_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.resample_interval = 50
        self.resample_prob = 0.2

        self.force_range = torch.tensor(force_range, device=self.device)
        self.offset_range = torch.tensor(offset_range, device=self.device)
        self.duration_range = torch.tensor(duration_range, device=self.device)

        self.arange = torch.arange(self.num_envs, device=self.device)

    def step(self, substep):
        arange = self.arange
        quat = self.asset.data.body_link_quat_w[arange, self.body_id]
        forces_b = quat_rotate_inverse(
            quat.reshape(self.num_envs, 4),
            self.force.get_force()
        )
        self.asset._external_force_b[arange, self.body_id] += forces_b
        self.asset._external_torque_b[arange, self.body_id] += self.force.offset.cross(forces_b, dim=-1)
        self.asset.has_external_wrench = True
    
    def reset(self, env_ids: torch.Tensor):
        self.force.duration.data[env_ids] = 0.
        
    def update(self):
        resample = (self.env.episode_length_buf % self.resample_interval == 0)
        expired = self.force.time > self.force.duration
        resample = resample & expired.squeeze(-1) & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        force = ConstantForce.sample(self.num_envs, self.force_range, self.offset_range, self.duration_range, self.device)
        self.force.time.add_(self.env.step_dt)
        self.force = force.where(resample, self.force)
        body_id = self.all_body_ids[torch.randint(0, len(self.all_body_ids), (self.num_envs,), device=self.device)]
        self.body_id = torch.where(resample, body_id, self.body_id)
    
    def debug_draw(self):
        self.env.debug_draw.vector(
            self.asset.data.body_link_pos_w[torch.arange(self.num_envs, device=self.device), self.body_id],
            self.force.get_force() /  9.81,
            color=(1.0, 0.6, 0.0, 1.0),
            size=3.0,
        )
        

def clamp_norm(x: torch.Tensor, min: float = 0.0, max: float = torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x


def random_scale(x: torch.Tensor, low: float, high: float, homogeneous: bool=False):
    if homogeneous:
        u = torch.rand(*x.shape[:1], 1, device=x.device)
    else:
        u = torch.rand_like(x)
    return x * (u * (high - low) + low), u

def random_shift(x: torch.Tensor, low: float, high: float):
    return x + x * (torch.rand_like(x) * (high - low) + low)

def sample_uniform(size, low: float, high: float, device: torch.device = "cpu"):
    return torch.rand(size, device=device) * (high - low) + low

def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)

def uniform_like(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(x)
    return low + r * (high - low)

def log_uniform(low: torch.Tensor, high: torch.Tensor):
    return uniform(low.log(), high.log()).exp()

def angle_mix(a: torch.Tensor, b: torch.Tensor, weight: float=0.1):
    d = a - b
    d[d > torch.pi] -= 2 * torch.pi
    d[d < -torch.pi] += 2 * torch.pi
    return a - d * weight
