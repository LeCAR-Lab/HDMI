from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.base import Randomization as BaseRandomization

import torch
from typing import Tuple, TYPE_CHECKING
from mjlab.third_party.isaaclab.isaaclab.utils.math import sample_uniform


RobotObjectTrackRandomization = BaseRandomization[RobotObjectTracking]

class object_body_randomization(RobotObjectTrackRandomization ):
    def __init__(
        self,
        dynamic_friction_range: Tuple[float, float]=(0.6, 1.0),
        restitution_range: Tuple[float, float]=(0.0, 0.2),
        mass_range: Tuple[float, float]=(1.0, 10.0),
        static_friction_range: Tuple[float, float] | None = None,
        static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Validate that only one of static_friction_range or static_dynamic_friction_ratio_range is specified
        if static_friction_range is not None and static_dynamic_friction_ratio_range is not None:
            raise ValueError("Cannot specify both static_friction_range and static_dynamic_friction_ratio_range")
        if static_friction_range is None and static_dynamic_friction_ratio_range is None:
            raise ValueError("Must specify either static_friction_range or static_dynamic_friction_ratio_range")
        
        self.object = self.command_manager.object

        self.mass_range = mass_range

        self.all_indices_cpu = torch.arange(self.env.scene.num_envs)

        # randomize all shapes of the object
        max_shapes = self.object.root_physx_view.max_shapes
        self.shape_ids = torch.arange(0, max_shapes) 

        self.num_buckets = 64
        
        # Sample dynamic friction and restitution buckets
        self.dynamic_friction_buckets = sample_uniform(*tuple(dynamic_friction_range), (self.num_buckets,), "cpu")
        self.restitution_buckets = sample_uniform(*tuple(restitution_range), (self.num_buckets,), "cpu")
        
        # Handle static friction based on which parameter is specified
        if static_friction_range is not None:
            self.static_friction_buckets = sample_uniform(*tuple(static_friction_range), (self.num_buckets,), "cpu")
        else:
            self.static_dynamic_friction_ratio_buckets = sample_uniform(*tuple(static_dynamic_friction_ratio_range), (self.num_buckets,), "cpu")

    def startup(self):
        masses = self.object.data.default_mass.clone()
        inertias = self.object.data.default_inertia.clone()
        new_masses = sample_uniform(*self.mass_range, masses.shape, "cpu")

        scale = new_masses / masses
        masses[:] *= scale
        if inertias.ndim == 2:
            inertias[:] *= scale
        elif inertias.ndim == 3:
            inertias[:] *= scale.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid shape for inertias: {inertias.shape}")
        self.object.root_physx_view.set_masses(masses, self.all_indices_cpu)
        self.object.root_physx_view.set_inertias(inertias, self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_masses(), masses, atol=1e-4)
        assert torch.allclose(self.object.root_physx_view.get_inertias(), inertias, atol=1e-4)

        materials = self.object.root_physx_view.get_material_properties().clone()
        shape = (self.object.num_instances, 1)
        dynamic_friction = self.dynamic_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        restitution = self.restitution_buckets[torch.randint(0, self.num_buckets, shape)]
        if hasattr(self, "static_friction_buckets"):
            static_friction = self.static_friction_buckets[torch.randint(0, self.num_buckets, shape)]
        else:
            static_friction_ratio = self.static_dynamic_friction_ratio_buckets[torch.randint(0, self.num_buckets, shape)]
            static_friction = dynamic_friction * static_friction_ratio
        materials[:, self.shape_ids, 0] = static_friction
        materials[:, self.shape_ids, 1] = dynamic_friction
        materials[:, self.shape_ids, 2] = restitution
        self.object.root_physx_view.set_material_properties(materials.flatten(), self.all_indices_cpu)
        assert torch.allclose(self.object.root_physx_view.get_material_properties(), materials, atol=1e-4)

class object_joint_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        friction_range: Tuple[float, float]=(0.0, 0.1),
        damping_range: Tuple[float, float]=(1.0, 10.0),
        armature_range: Tuple[float, float]=(0.0, 0.02),
        **kwargs
    ):
        super().__init__(**kwargs)
        if TYPE_CHECKING:
            from active_adaptation.assets.objects_mjlab import CustomEntity as CustomArticulation
        self.object: CustomArticulation = self.command_manager.object
        self.friction_range = friction_range
        self.damping_range = damping_range
        self.armature_range = armature_range

        self.joint_id_asset = 0
    
    def startup(self):
        return
        # door_armature = sample_uniform(*self.armature_range, (self.object.num_instances, 1), self.device)
        # self.object.write_joint_armature_to_sim(door_armature, joint_ids=[self.joint_id_asset])

    def reset(self, env_ids: torch.Tensor):
        joint_friction = sample_uniform(*self.friction_range, (len(env_ids),), self.device)
        joint_damping = sample_uniform(*self.damping_range, (len(env_ids),), self.device)

        self.object._custom_friction[env_ids] = joint_friction
        self.object._custom_damping[env_ids] = joint_damping
