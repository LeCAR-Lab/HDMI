import os
import json
import torch

import active_adaptation
from active_adaptation.envs.base import _Env

class SimpleEnv(_Env):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.robot = self.scene.entities["robot"]
        
        self.action_buf: torch.Tensor = self.action_manager.action_buf
        self.last_action: torch.Tensor = self.action_manager.applied_action

    def setup_scene(self):
        if active_adaptation.get_backend() == "isaac":
            import isaaclab.sim as sim_utils
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import AssetBaseCfg, ArticulationCfg
            from isaaclab.sensors import ContactSensorCfg
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
            from isaaclab.utils import configclass
            from active_adaptation.assets import ROBOTS, OBJECTS, get_asset_meta
            from active_adaptation.envs.terrain import TERRAINS
            
            env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
            scene_cfg = InteractiveSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing, replicate_physics=False)
            scene_cfg.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
            scene_cfg.robot: ArticulationCfg = ROBOTS[self.cfg.robot.name]
            
            if hasattr(self.cfg.robot, 'override_params'):
                from active_adaptation.utils import update_class_from_dict
                update_class_from_dict(scene_cfg.robot, self.cfg.robot.override_params, _ns="")
            
            scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
            robot_type = self.cfg.robot.get("robot_type", self.cfg.robot.name)
            scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path.format(ROBOT_TYPE=robot_type)

            # if self.cfg.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking":
            if "object_asset_name" in self.cfg.command:
                extra_object_names = self.cfg.command.get("extra_object_names", [])
                for extra_obj_name in extra_object_names:
                    extra_obj_cfg = OBJECTS[extra_obj_name]
                    extra_obj_cfg.prim_path = "{ENV_REGEX_NS}/" + extra_obj_name
                    setattr(scene_cfg, extra_obj_name, extra_obj_cfg)

                obj_name = self.cfg.command.object_asset_name
                obj_contact_body_name = self.cfg.command.object_body_name

                obj_cfg = OBJECTS[obj_name]
                obj_cfg.prim_path = "{ENV_REGEX_NS}/" + obj_name
                obj_type = self.cfg.command.get("object_type", obj_name)
                obj_cfg.spawn.usd_path = obj_cfg.spawn.usd_path.format(OBJECT_TYPE=obj_type)
                print(f"Using object type {obj_type} with asset {obj_cfg.spawn.usd_path}")
                setattr(scene_cfg, obj_name, obj_cfg)

                # add contact sensor to the box
                eef_names = self.cfg.command.get("contact_eef_body_name", [])
                contact_geom_prim_path = "{ENV_REGEX_NS}/" + obj_name + "/" + obj_contact_body_name

                for eef_name in eef_names:
                    contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"
                    eef_prim_path = "{ENV_REGEX_NS}/Robot/" + eef_name
                    setattr(scene_cfg, contact_sensor_name, ContactSensorCfg(
                        prim_path=eef_prim_path,
                        history_length=0,
                        track_air_time=False,
                        filter_prim_paths_expr=[contact_geom_prim_path],
                    ))
                    
            body_scale_rand = self.cfg.randomization.get("body_scale", None)
            if body_scale_rand is not None:
                from active_adaptation.assets.spawn import clone
                asset = getattr(scene_cfg, body_scale_rand.name)
                spawn_func = asset.spawn.func.__wrapped__
                asset.spawn.func = clone(spawn_func)
                asset.spawn.scale_range = tuple(body_scale_rand.scale_range)
                asset.spawn.homogeneous_scale = body_scale_rand.get("homogeneous_scale", False)
                print(f"Randomized {body_scale_rand.name} scale to {asset.spawn.scale_range}")

            scene_cfg.terrain = TERRAINS[self.cfg.terrain]
            scene_cfg.contact_forces = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|wrist_.*)_link", 
                history_length=3,
                track_air_time=True
            )

            if self.cfg.get("enable_cameras", False):
                from isaaclab.sensors import TiledCameraCfg
                # this is from my ros reading
                # camera_spawn_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                #     intrinsic_matrix=[
                #         390.2486572265625, 0.0, 324.02740478515625,
                #         0.0, 390.2486572265625, 234.0133056640625,
                #         0.0, 0.0, 1.0,
                #     ],
                #     width=640,
                #     height=480,
                #     clipping_range=(0.1, 4.0),
                #     focal_length=None,
                #     focus_distance=400.0,
                #     f_stop=0.0,
                #     projection_type="pinhole",
                #     lock_camera=True,
                # )
                # from https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/27498c09159a99879a882e212059988df86018c8/tasks/common_config/camera_configs.py#L21
                camera_spawn_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=7.6,
                    focus_distance=400.0,
                    horizontal_aperture=20.0,
                    clipping_range=(0.1, 1.0e5),
                )
                tiled_camera: TiledCameraCfg = TiledCameraCfg(
                    prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",
                    spawn=camera_spawn_cfg,
                    offset=TiledCameraCfg.OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(0.5, -0.5, 0.5, -0.5),
                        convention="ros"
                    ),
                    # NOTE: remove rgb does not improve speed, and only slightly reduces memory usage
                    # TODO: depth or distance_to_image_plane?
                    data_types=["rgb", "depth", "distance_to_image_plane"],
                    update_latest_camera_pose=True,
                    update_period=0.02,
                    width=self.cfg.camera_width,
                    height=self.cfg.camera_height,
                )
                scene_cfg.tiled_camera = tiled_camera
            
            # if self.cfg.get("enable_raycaster", False):
            #     from isaaclab.sensors import RayCasterCfg
            #     raycaster = RayCasterCfg(
            #         prim_path="/World/envs/env_.*/Robot/d435_link/front_cam",
            #         update_period=0.02,
            #         offset=RayCasterCfg.OffsetCfg(
            #             pos=(0.0, 0.0, 0.0),
            #             rot=(0.5, -0.5, 0.5, -0.5),
            #         ),
            #     )
            #     scene_cfg.raycaster = raycaster
            
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.cfg.sim.isaac_physics_dt,
                render=sim_utils.RenderCfg(
                    rendering_mode="quality",
                    # antialiasing_mode="FXAA",
                    # enable_global_illumination=True,
                    # enable_reflections=True,
                ),
                device=f"cuda:{active_adaptation.get_local_rank()}"
            )
            
            # slightly reduces GPU memory usage
            # sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
            # sim_cfg.physx.gpu_max_rigid_patch_count = 2**21
            sim_cfg.physx.gpu_found_lost_pairs_capacity = 2538320 # 2**20
            sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 61999079 # 2**26
            sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
            sim_cfg.physx.enable_stabilization = False
            # sim_cfg.physx.gpu_collision_stack_size = 2**25
            # sim_cfg.physx.gpu_heap_capacity = 2**24
            
            self.sim, self.scene = scene.create_isaaclab_sim_and_scene(sim_cfg, scene_cfg)

            # set camera view for "/OmniverseKit_Persp" camera
            self.sim.set_camera_view(eye=self.cfg.viewer.eye, target=self.cfg.viewer.lookat)
            try:
                import omni.replicator.core as rep
                # create render product
                self._render_product = rep.create.render_product(
                    "/OmniverseKit_Persp", tuple(self.cfg.viewer.resolution)
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
                # self._seg_annotator = rep.AnnotatorRegistry.get_annotator(
                #     "instance_id_segmentation_fast", 
                #     device="cpu",
                # )
                # self._seg_annotator.attach([self._render_product])
                # for _ in range(4):
                #     self.sim.render()
            except ModuleNotFoundError as e:
                print("Set app.enable_cameras=true to use cameras.")
            
            try:
                from active_adaptation.utils.debug import DebugDraw
                self.debug_draw = DebugDraw()
                print("[INFO] Debug Draw API enabled.")
            except ModuleNotFoundError:
                print()
            
            asset_meta = get_asset_meta(self.scene["robot"])
            path = os.path.join(os.getcwd(), "asset_meta.json")
            print(f"Saving asset meta to {path}")
            with open(path, "w") as f:
                json.dump(asset_meta, f, indent=4)
        elif active_adaptation.get_backend() == "mujoco":
            from active_adaptation.envs.mujoco import MJScene, MJSim
            from active_adaptation.assets_mjcf import ROBOTS

            @configclass
            class SceneCfg:
                robot = ROBOTS[self.cfg.robot.name]
                contact_forces = "robot"
            
            self.scene = MJScene(SceneCfg())
            self.sim = MJSim(self.scene)
        elif active_adaptation.get_backend() == "mjlab":
            from mjlab.scene import SceneCfg as MJSceneCfg

            env_spacing = self.cfg.viewer.get("env_spacing", 2.0)
            scene_cfg = MJSceneCfg(num_envs=self.cfg.num_envs, env_spacing=env_spacing)
            
            from mjlab.terrains.terrain_importer import TerrainImporterCfg
            terrain_cfg = TerrainImporterCfg(terrain_type="plane", env_spacing=env_spacing, num_envs=self.cfg.num_envs)
            scene_cfg.terrain = terrain_cfg
            
            # TODO: support different robots
            # from mjlab.entity import EntityCfg
            # robot_cfg = EntityCfg()
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
            robot_cfg = G1_ROBOT_CFG
            
            
            from mjlab.utils.spec_config import ContactSensorCfg as MJContactCfg
            left_foot_contact_sensor_cfg = MJContactCfg(
                name="left_foot_self_collision",
                body1="left_ankle_roll_link",
                data=("force",),
                reduce="netforce",
                num=10,
            )
            right_foot_contact_sensor_cfg = MJContactCfg(
                name="right_foot_self_collision",
                body1="right_ankle_roll_link",
                data=("force",),
                reduce="netforce",
                num=10,
            )
            
            scene_cfg.entities["robot"] = robot_cfg
            
            if "object_asset_name" in self.cfg.command:
                from active_adaptation.assets.objects_mjlab import OBJECTS_MJLAB
                obj_name = self.cfg.command.object_asset_name
                obj_cfg = OBJECTS_MJLAB[obj_name]

                # TODO: now mjlab does not support body1 and body2 in different entities
                # # add contact sensors to the object
                # eef_names = self.cfg.command.get("contact_eef_body_name", [])
                # obj_body_name = self.cfg.command.object_body_name
                # contact_sensors = []
                # for eef_name in eef_names:
                #     contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"
                #     contact_sensor_cfg = MJContactCfg(
                #         name=contact_sensor_name,
                #         body1=f"{obj_name}/{obj_body_name}",
                #         body2=f"robot/{eef_name}",
                #         data=("force",),
                #         reduce="netforce",
                #         num=4,
                #     )
                #     contact_sensors.append(contact_sensor_cfg)
                # obj_cfg.sensors = tuple(contact_sensors)

                scene_cfg.entities[obj_name] = obj_cfg
            
                # add contact sensors to the scene
                eef_names = self.cfg.command.get("contact_eef_body_name", [])
                obj_body_name = self.cfg.command.object_body_name
                contact_sensor_cfgs = []
                for eef_name in eef_names:
                    contact_sensor_name = f"{eef_name}_{obj_name}_contact_forces"
                    contact_sensor_cfg = MJContactCfg(
                        name=contact_sensor_name,
                        body1=f"{obj_name}/{obj_body_name}",
                        body2=f"robot/{eef_name}",
                        data=("force",),
                        reduce="netforce",
                        num=4,
                    )
                    contact_sensor_cfgs.append(contact_sensor_cfg)
                scene_cfg.sensors = tuple(contact_sensor_cfgs)

            from mjlab.sim import MujocoCfg, SimulationCfg as MJSimCfg
            self.sim_cfg = sim_cfg = MJSimCfg(
                nconmax=150_000,
                njmax=250,
                mujoco=MujocoCfg(
                    timestep=self.cfg.sim.mjlab_physics_dt,
                    iterations=10,
                    ls_iterations=20,
                ),
            )
            
            from mjlab.scene import Scene
            from mjlab.sim.sim import Simulation
            self.scene = Scene(scene_cfg, device=self.device)
            sim_cfg.mujoco.edit_spec(self.scene.spec)
            print(f"[INFO]: Scene manager: {self.scene}")

            self.sim = Simulation(
                num_envs=self.scene.num_envs,
                cfg=sim_cfg,
                model=self.scene.compile(),
                device=self.device,
            )
                    
            self.scene.initialize(
                mj_model=self.sim.mj_model,
                model=self.sim.model,
                data=self.sim.data,
            )

        
    def _reset_idx(self, env_ids: torch.Tensor):
        init_root_state = self.command_manager.sample_init(env_ids)
        if init_root_state is not None and not self.robot.is_fixed_base:
            self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )
        self.stats[env_ids] = 0.

    def render(self, mode: str="human"):
        # look_at_env_id = self.lookat_env_i
        # self.sim.set_camera_view(
        #     eye=self.robot.data.root_link_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.eye),
        #     target=self.robot.data.root_link_pos_w[look_at_env_id].cpu() + torch.as_tensor(self.cfg.viewer.lookat)
        # )
        return super().render(mode)

