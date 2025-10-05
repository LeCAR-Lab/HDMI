import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import warnings
from pathlib import Path
from typing import Sequence, Union, Any, Dict
from dataclasses import dataclass, replace

from isaaclab.utils import string as string_utils
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.envs.base import _Env
from active_adaptation.utils.math import quat_rotate_inverse, quat_rotate
from tensordict import TensorClass


ArrayType = Union[np.ndarray, torch.Tensor]


@dataclass
class MJArticulationCfg:
    mjcf_path: str
    init_state: Any
    actuators: Dict
    body_names_isaac: Sequence[str]
    joint_names_isaac: Sequence[str]
    joint_symmetry_mapping: Dict=None
    spatial_symmetry_mapping: Dict=None


@dataclass
class MJArticulationData:
    default_joint_pos: ArrayType
    default_joint_vel: ArrayType
    default_root_state: ArrayType
    default_mass: ArrayType
    default_inertia: ArrayType
    
    joint_stiffness: ArrayType = None
    joint_damping: ArrayType = None

    body_link_pos_w: ArrayType = None
    body_link_quat_w: ArrayType = None
    
    joint_pos: ArrayType = None
    joint_pos_target: ArrayType = None
    
    joint_vel: ArrayType = None
    joint_vel_target: ArrayType = None

    applied_torque: ArrayType = None
    projected_gravity_b: ArrayType = None
    
    body_com_vel_w: ArrayType = None
    # body_link_lin_vel_w: ArrayType = None
    # body_link_ang_vel_w: ArrayType = None
    root_com_lin_vel_w: ArrayType = None
    root_com_ang_vel_w: ArrayType = None
    root_ang_vel_b: ArrayType = None
    root_lin_vel_b: ArrayType = None
    heading_w: ArrayType = None

    @property
    def body_link_lin_vel_w(self):
        return self.body_com_vel_w[..., 3:]
    
    @property
    def body_link_ang_vel_w(self):
        return self.body_com_vel_w[..., :3]

    @property
    def root_link_pos_w(self):
        return self.body_link_pos_w[..., 0, :]
    
    @property
    def root_link_quat_w(self):
        return self.body_link_quat_w[..., 0, :]
    
    # @property
    # def root_com_lin_vel_w(self):
    #     return self.body_com_vel_w[..., 0, :3]
    
    # @property
    # def root_com_ang_vel_w(self):
    #     return self.body_com_vel_w[..., 0, 3:]
    
    @property
    def root_state_w(self):
        return torch.cat([self.body_link_pos_w[:, 0, :], self.body_link_quat_w[:, 0, :]], dim=-1)


class MJPhysicsView:
    def __init__(self, articulation: "MJArticulation"):
        self.articulation = articulation


class MJArticulation:
    
    num_instances = 1
    is_fixed_base = False

    def __init__(self, cfg: MJArticulationCfg):
        self.cfg = cfg
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.body_names_isaac = list(cfg.body_names_isaac)
        self.body_names_mjc = []
        body_adrs = []
        for i in range(1, self.mj_model.nbody): # skip the world body
            body = self.mj_model.body(i)
            self.body_names_mjc.append(body.name)
            body_adrs.append(i)
        
        if not set(self.body_names_isaac) == set(self.body_names_mjc):
            warnings.warn(
                f"Isaac body names do not match mujoco body names:\n"
                f"Isaac - Mujoco: {set(self.body_names_isaac) - set(self.body_names_mjc)}\n"
                f"Mujoco - Isaac: {set(self.body_names_mjc) - set(self.body_names_isaac)}\n",
                category=UserWarning
            )

        # find only the actuated joints
        self.joint_names_isaac = list(cfg.joint_names_isaac)
        self.joint_names_mjc = []

        joint_qposadr = []
        joint_qveladr = []
        for i in range(self.mj_model.nu):
            actuator = self.mj_model.actuator(i)
            if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_id = actuator.trnid[0]
                joint = self.mj_model.joint(actuator.trnid[0])
                self.joint_names_mjc.append(joint.name)
                joint_qposadr.append(self.mj_model.jnt_qposadr[joint_id])
                joint_qveladr.append(self.mj_model.jnt_dofadr[joint_id])
        
        if not set(self.joint_names_isaac) == set(self.joint_names_mjc):
            warnings.warn(
                f"Isaac joint names do not match mujoco joint names:\n"
                f"Isaac - Mujoco: {set(self.joint_names_isaac) - set(self.joint_names_mjc)}\n"
                f"Mujoco - Isaac: {set(self.joint_names_mjc) - set(self.joint_names_isaac)}\n",
                category=UserWarning
            )
        
        # Isaac assets may have less joints/bodies due to asset simplification
        self._jnt_isaac2mjc = [self.joint_names_isaac.index(joint_name) for joint_name in self.joint_names_mjc if joint_name in self.joint_names_isaac]
        self._jnt_mjc2isaac = [self.joint_names_mjc.index(joint_name) for joint_name in self.joint_names_isaac]
        self._body_isaac2mjc = [self.body_names_isaac.index(body_name) for body_name in self.body_names_mjc if body_name in self.body_names_isaac]
        self._body_mjc2isaac = [self.body_names_mjc.index(body_name) for body_name in self.body_names_isaac]
        
        self.body_adrs = np.array(body_adrs)
        self.joint_qposadr = np.array(joint_qposadr)
        self.joint_qveladr = np.array(joint_qveladr)
        
        # read/write mujoco data in isaac order
        self.body_adrs_read = self.body_adrs[self._body_mjc2isaac]
        self.body_adrs_write = self.body_adrs[self._body_isaac2mjc]
        self.joint_qposadr_read = self.joint_qposadr[self._jnt_mjc2isaac]
        self.joint_qveladr_read = self.joint_qveladr[self._jnt_mjc2isaac]
        self.joint_qposadr_write = self.joint_qposadr[self._jnt_isaac2mjc]
        self.joint_qveladr_write = self.joint_qveladr[self._jnt_isaac2mjc]

        joint_ids, joint_names, joint_pos = string_utils.resolve_matching_names_values(self.cfg.init_state["joint_pos"], self.joint_names_isaac)
        if len(joint_names) < len(self.joint_names_isaac):
            print(f"Missing joint names: {set(self.joint_names_isaac) - set(joint_names)}")
        default_joint_pos = torch.zeros(self.num_joints)
        default_joint_pos[joint_ids] = torch.as_tensor(joint_pos)
        for jname, jpos in zip(self.joint_names, default_joint_pos, strict=True):
            print(jname, jpos)
        default_joint_vel = torch.zeros(self.num_joints)

        joint_stiffness = torch.zeros(self.num_joints)
        joint_damping = torch.zeros(self.num_joints)
        
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            ids, _, values = string_utils.resolve_matching_names_values(actuator_cfg["stiffness"], self.joint_names_isaac)
            joint_stiffness[ids] = torch.as_tensor(values)
            ids, _, values = string_utils.resolve_matching_names_values(actuator_cfg["damping"], self.joint_names_isaac)
            joint_damping[ids] = torch.as_tensor(values)

        diag_inertia = torch.as_tensor(self.mj_model.body_inertia[self.body_adrs], dtype=torch.float32)
        self._data = MJArticulationData(
            default_joint_pos=default_joint_pos[None],
            default_joint_vel=default_joint_vel[None],
            default_root_state=torch.tensor([[*cfg.init_state["pos"], 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]),
            default_mass=torch.as_tensor(self.mj_model.body_mass[self.body_adrs], dtype=torch.float32)[None],
            default_inertia=diag_inertia.diag_embed().flatten(1)[None],
            joint_stiffness=joint_stiffness[None],
            joint_damping=joint_damping[None],
            applied_torque=torch.zeros(1, self.num_joints),
            # batch_size=[1]
        )
        self._data.joint_pos_target = self._data.default_joint_pos.clone()
        self._data.joint_vel_target = self._data.default_joint_vel.clone()
        
        self._external_force_b = torch.zeros(1, self.num_bodies, 3)
        self._external_torque_b = torch.zeros(1, self.num_bodies, 3)
        self.has_external_wrench = False
        
        self.timestamp = 0.
        
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.update(0.0)
    
    @property
    def joint_names(self):
        return self.joint_names_isaac
    
    @property
    def body_names(self):
        return self.body_names_isaac

    @property
    def num_joints(self):
        return len(self.joint_names)
    
    @property
    def num_bodies(self):
        return len(self.body_names)
    
    @property
    def data(self):
        return self._data

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.lab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names_isaac, preserve_order)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`omni.isaac.lab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.joint_names_isaac
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def update(self, dt: float):
        jpos = self.mj_data.qpos[self.joint_qposadr_read]
        jvel = self.mj_data.qvel[self.joint_qveladr_read]
        body_link_pos_w = self.mj_data.xpos[self.body_adrs_read]
        # body_link_ang_vel_w = self.mj_data.cvel[self.body_adrs_read, :3]
        # body_link_lin_vel_w = self.mj_data.cvel[self.body_adrs_read, 3:]
        body_com_vel_w = self.mj_data.cvel[self.body_adrs_read]
        body_link_quat_w = self.mj_data.xquat[self.body_adrs_read] # wxyz
        
        # rot = sRot.from_quat(self.mj_data.qpos[3:7], scalar_first=True)
        rot = sRot.from_quat(body_link_quat_w[0], scalar_first=True)
        projected_gravity_b = rot \
            .inv() \
            .apply(np.array([0., 0., -1.]))
        heading_w = rot.as_euler('xyz', degrees=False)[2]

        self._data = replace(
            self._data,
            body_link_pos_w=torch.as_tensor(body_link_pos_w, dtype=torch.float32)[None],
            body_link_quat_w=torch.as_tensor(body_link_quat_w, dtype=torch.float32)[None],
            # body_link_lin_vel_w=torch.as_tensor(body_link_lin_vel_w, dtype=torch.float32)[None],
            # body_link_ang_vel_w=torch.as_tensor(body_link_ang_vel_w, dtype=torch.float32)[None],
            body_com_vel_w=torch.as_tensor(body_com_vel_w, dtype=torch.float32)[None],
            joint_pos=torch.as_tensor(jpos, dtype=torch.float32)[None],
            joint_pos_target=self._data.joint_pos_target.clone(),
            joint_vel=torch.as_tensor(jvel, dtype=torch.float32)[None],
            joint_vel_target=self._data.joint_vel_target.clone(),
            projected_gravity_b=torch.as_tensor(projected_gravity_b, dtype=torch.float32)[None],
            heading_w=torch.as_tensor(heading_w, dtype=torch.float32)[None],
        )
        # self._data.root_com_lin_vel_w = torch.as_tensor(self.mj_data.qvel[:3], dtype=torch.float32)[None]
        # self._data.root_com_ang_vel_w = torch.as_tensor(self.mj_data.qvel[3:6], dtype=torch.float32)[None]
        self._data.root_com_lin_vel_w = self._data.body_link_lin_vel_w[:, 0]
        self._data.root_com_ang_vel_w = self._data.body_link_ang_vel_w[:, 0]
        self._data.root_ang_vel_b = quat_rotate_inverse(self._data.root_link_quat_w, self._data.root_com_ang_vel_w)
        self._data.root_lin_vel_b = quat_rotate_inverse(self._data.root_link_quat_w, self._data.root_com_lin_vel_w)
        
        if hasattr(self, "_log_path"):
            self._log_states.append(self._data)

    def write_root_state_to_sim(self, root_state: ArrayType, env_ids: ArrayType=None):
        self.mj_data.qpos[:3] = root_state[0, :3]
        self.mj_data.qpos[3:7] = root_state[0, 3:7]
        self.mj_data.qvel[:6] = 0.
        self.write_joint_state_to_sim(self.data.default_joint_pos, self.data.default_joint_vel, slice(None))
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def set_joint_position_target(self, target: ArrayType, joint_ids: ArrayType=None):
        if joint_ids is None:
            self._data.joint_pos_target[0] = target
        else:
            self._data.joint_pos_target[0, joint_ids] = target
    
    def set_joint_velocity_target(self, target: ArrayType, joint_ids: ArrayType=None):
        if joint_ids is None:
            self._data.joint_vel_target[0] = target
        else:
            self._data.joint_vel_target[0, joint_ids] = target
    
    def write_data_to_sim(self):
        pos_error = self._data.joint_pos_target - self.mj_data.qpos[None, self.joint_qposadr_read]
        vel_error = self._data.joint_vel_target - self.mj_data.qvel[None, self.joint_qveladr_read]
        
        torque = (self._data.joint_stiffness * pos_error + self._data.joint_damping * vel_error)
        self._data.applied_torque = torque.float()

        self.mj_data.ctrl[self._jnt_mjc2isaac] = torque[0]

        if self.has_external_wrench:
            self.mj_data.xfrc_applied[self.body_adrs_write, :3] = quat_rotate(self._data.root_link_quat_w, self._external_force_b)[0]
            self.mj_data.xfrc_applied[self.body_adrs_write, 3:] = quat_rotate(self._data.root_link_quat_w, self._external_torque_b)[0]

    def write_joint_state_to_sim(self, joint_pos: ArrayType, joint_vel: ArrayType, joint_ids: ArrayType=None, env_ids: ArrayType=None):
        if joint_ids is None:
            joint_ids = slice(None)
        if joint_pos is not None:
            joint_pos_all = self._data.joint_pos[0].clone()
            joint_pos_all[joint_ids] = joint_pos[0]
            self.mj_data.qpos[self.joint_qposadr_read] = joint_pos_all
        if joint_vel is not None:
            joint_vel_all = self._data.joint_vel[0].clone()
            joint_vel_all[joint_ids] = joint_vel[0]
            self.mj_data.qvel[self.joint_qveladr_read] = joint_vel_all

    def setup_logger(self, name: str):
        self._log_path = Path.cwd() / f"{name}.pt"
        self._log_states = []


@dataclass
class MjContactData:
    net_forces_w: ArrayType = None


class MjContactSensor:
    def __init__(self, articulation: MJArticulation):
        self.articulation = articulation
        self.body_names = self.articulation.body_names
        self.body_adrs_read = self.articulation.body_adrs_read
        self._data = MjContactData(
            net_forces_w=torch.zeros(1, self.articulation.num_bodies, 3)
        )
    
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False):
        return self.articulation.find_bodies(name_keys, preserve_order)

    def update(self, dt: float):
        cfrc_ext = self.articulation.mj_data.cfrc_ext[self.body_adrs_read, :3]
        self._data.net_forces_w = torch.as_tensor(cfrc_ext, dtype=torch.float32)[None]

    @property
    def data(self):
        return self._data


class MJScene:
    num_envs = 1

    def __init__(self, cfg):
        self.cfg = cfg
        self.articulations = {}
        self.sensors = {}
        
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            print(asset_name, asset_cfg)
            if isinstance(asset_cfg, MJArticulationCfg):
                articulation = MJArticulation(asset_cfg)
                articulation.setup_logger(asset_name)
                self.articulations[asset_name] = articulation
            elif isinstance(asset_cfg, str):
                sensor = MjContactSensor(self.articulations[asset_cfg])
                self.sensors[asset_name] = sensor

        self.viewer = mujoco.viewer.launch_passive(self.articulations["robot"].mj_model, self.articulations["robot"].mj_data, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1

        self.mj_model = self.articulations["robot"].mj_model
        self.mj_data = self.articulations["robot"].mj_data
        self.env_origins = torch.zeros(1, 3)

    def reset(self, env_ids: torch.Tensor):
        for articulation in self.articulations.values():
            continue
            articulation.reset(env_ids)

    def update(self, dt: float):
        for articulation in self.articulations.values():
            articulation.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt)
    
    def write_data_to_sim(self):
        for articulation in self.articulations.values():
            articulation.write_data_to_sim()

    def __getitem__(self, key: str):
        result = self.articulations.get(key)
        result = result or self.sensors.get(key)
        return result

    def create_arrow_marker(self, radius: float, rgba):
        scene = self.viewer.user_scn
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([radius, radius, 1.0], dtype=np.float64),
            pos=np.zeros(3),
            mat=sRot.random().as_matrix().reshape(-1),
            rgba=np.array(rgba, dtype=np.float64),
        )
        return MjvGeom(scene.geoms[scene.ngeom - 1])
    
    def create_sphere_marker(self, radius: float, rgba):
        scene = self.viewer.user_scn
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, radius, radius], dtype=np.float64),
            pos=np.zeros(3),
            mat=sRot.random().as_matrix().reshape(-1),
            rgba=np.array(rgba, dtype=np.float64),
        )
        return MjvGeom(scene.geoms[scene.ngeom - 1])


    def close(self):
        for articulation in self.articulations.values():
            path = articulation._log_path
            states = torch.stack(articulation._log_states)
            torch.save(states.__dict__, path)


class MJSim:
    
    device = "cpu"

    def __init__(self, scene: MJScene):
        self.scene = scene
        self.mj_model = scene.mj_model
        self.mj_data = scene.mj_data
    
    def render(self):
        self.scene.viewer.sync()

    def get_physics_dt(self):
        return self.mj_model.opt.timestep

    def has_gui(self):
        return True

    def step(self, render: bool=False):
        mujoco.mj_step(self.mj_model, self.mj_data)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        time.sleep(self.get_physics_dt())


class MjvGeom:
    def __init__(self, geom):
        self.geom: mujoco.MjvGeom = geom

    def from_to(self, from_, to):
        mujoco.mjv_connector(
            self.geom,
            self.geom.type,
            width=0.05,
            from_=np.array(from_.reshape(3)).astype(np.float64),
            to=np.array(to.reshape(3)).astype(np.float64),
        )
