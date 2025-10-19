"""This is a minimal example of how to read the retargeted data and rollout in HDMI environment (mjlab version)."""

import os
import time

import active_adaptation
import cv2
import mujoco
import numpy as np
import torch
import warp as wp
from active_adaptation.envs import SimpleEnv
from omegaconf import OmegaConf
from tensordict import TensorDict


def setup_env(task: str = "move_suitcase", max_episode_length: int = 1000):
    """Setup and reset a single HDMI environment for visualization.

    Args:
        task: Task name (e.g., "move_suitcase")
        max_episode_length: Maximum episode length

    Returns:
        SimpleEnv instance
    """
    # Use active_adaptation module path to find HDMI directory
    hdmi_dir = os.path.dirname(os.path.dirname(active_adaptation.__file__))

    # Load base config
    base_config_path = os.path.join(hdmi_dir, "cfg/task/base/hdmi-base.yaml")
    base_cfg = OmegaConf.load(base_config_path)
    base_cfg.max_episode_length = max_episode_length

    # Load task-specific config
    task_config_path = os.path.join(hdmi_dir, f"cfg/task/G1/hdmi/{task}.yaml")
    task_cfg = OmegaConf.load(task_config_path)

    # Merge configurations (task overrides base)
    cfg = OmegaConf.merge(base_cfg, task_cfg)

    # Override with custom parameters for single environment visualization
    cfg.num_envs = 16  # Single environment
    cfg.viewer.headless = False  # Enable visualization
    cfg.viewer.env_spacing = 0.0

    # Disable struct mode to allow modifications
    OmegaConf.set_struct(cfg, False)

    # Remove observation groups that are not needed
    for obs_group_key in list(cfg.observation.keys()):
        if obs_group_key not in [
            "command",
            "policy",
            "priv",
        ] and not obs_group_key.endswith("_"):
            cfg.observation.pop(obs_group_key)

    # Remove randomizations in command manager
    if "pose_range" in cfg.command:
        for key in cfg.command.pose_range:
            cfg.command.pose_range[key] = [0.0, 0.0]
    if "velocity_range" in cfg.command:
        for key in cfg.command.velocity_range:
            cfg.command.velocity_range[key] = [0.0, 0.0]
    if "object_pose_range" in cfg.command:
        for key in cfg.command.object_pose_range:
            cfg.command.object_pose_range[key] = [0.0, 0.0]
    if "init_joint_pos_noise" in cfg.command:
        cfg.command.init_joint_pos_noise = 0.0
    if "init_joint_vel_noise" in cfg.command:
        cfg.command.init_joint_vel_noise = 0.0

    # Disable motion sampling for deterministic resets
    cfg.command.sample_motion = False
    cfg.command.reset_range = None

    # Disable action delay and alpha randomization
    if "action" in cfg:
        cfg.action.min_delay = 0
        cfg.action.max_delay = 0
        cfg.action.alpha = [1.0, 1.0]

    # Remove randomizations - set all ranges to their mean
    for rand_key, rand_params in cfg.randomization.items():
        if rand_params is not None and isinstance(rand_params, dict):
            for param_key, param_value in rand_params.items():
                if isinstance(param_value, (list, tuple)) and len(param_value) == 2:
                    mean_val = (param_value[0] + param_value[1]) / 2.0
                    cfg.randomization[rand_key][param_key] = [mean_val, mean_val]
                elif isinstance(param_value, dict):
                    for sub_key, sub_value in param_value.items():
                        if isinstance(sub_value, (list, tuple)) and len(sub_value) == 2:
                            mean_val = (sub_value[0] + sub_value[1]) / 2.0
                            cfg.randomization[rand_key][param_key][sub_key] = [
                                mean_val,
                                mean_val,
                            ]

    # Create environment
    env = SimpleEnv(cfg)

    # Set environment to eval mode for deterministic resets
    env.eval()
    env.reset()

    return env


def set_to_env(env, qpos, qvel, ctrl):
    """Set qpos, qvel, and ctrl to the environment.

    Args:
        env: HDMI environment
        qpos: Joint positions (1D tensor)
        qvel: Joint velocities (1D tensor)
        ctrl: Control inputs (1D tensor)
    """
    qpos_wp = wp.from_torch(qpos.repeat(env.num_envs, 1).float().to(env.device))
    qvel_wp = wp.from_torch(qvel.repeat(env.num_envs, 1).float().to(env.device))
    ctrl_wp = wp.from_torch(ctrl.repeat(env.num_envs, 1).float().to(env.device))
    wp.copy(env.sim.wp_data.qpos, qpos_wp)
    wp.copy(env.sim.wp_data.qvel, qvel_wp)
    wp.copy(env.sim.wp_data.ctrl, ctrl_wp)


def setup_renderer(mj_model: mujoco.MjModel, width: int = 720, height: int = 480):
    """Setup MuJoCo renderer for saving videos.

    Args:
        mj_model: MuJoCo model
        width: Image width
        height: Image height

    Returns:
        MuJoCo Renderer instance
    """
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    return renderer


def render_image(
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
):
    """Render side-by-side comparison of reference and simulation.

    Args:
        renderer: MuJoCo renderer
        mj_model: MuJoCo model
        mj_data: Current simulation data
        mj_data_ref: Reference motion data

    Returns:
        Concatenated image with ref on left, sim on right
    """
    # Render sim
    mujoco.mj_kinematics(mj_model, mj_data)
    try:
        renderer.update_scene(mj_data, "front")
    except Exception:
        renderer.update_scene(mj_data, 0)
    sim_image = renderer.render()
    cv2.putText(
        sim_image,
        "sim",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # Render ref
    mujoco.mj_forward(mj_model, mj_data_ref)
    try:
        renderer.update_scene(mj_data_ref, "front")
    except Exception:
        renderer.update_scene(mj_data_ref, 0)
    ref_image = renderer.render()
    cv2.putText(
        ref_image,
        "ref",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    image = np.concatenate([ref_image, sim_image], axis=1)
    return image


def main(
    trajectory_path: str = "trajectory_hdmi.npz",
    task: str = "move_suitcase",
):
    """Rollout control trajectory in HDMI environment.

    Args:
        trajectory_path: Path to the trajectory file
        task: Task name for environment setup
    """
    # Load trajectory data
    print(f"Loading trajectory from: {trajectory_path}")
    raw_data = np.load(trajectory_path)
    ctrl = raw_data["ctrl"]
    qpos = raw_data["qpos"]
    qvel = raw_data["qvel"]
    ctrl = ctrl.reshape(-1, ctrl.shape[-1])
    qpos = qpos.reshape(-1, qpos.shape[-1])
    qvel = qvel.reshape(-1, qvel.shape[-1])

    print(f"Loaded control sequence with shape: {ctrl.shape}")

    # Setup environment
    print(f"Setting up HDMI environment for task: {task}")
    # setup env
    env = setup_env(task=task, max_episode_length=ctrl.shape[0])
    # Set initial qpos, qvel and ctrl
    device = env.device
    ctrl_torch = torch.from_numpy(ctrl).float().to(device)
    qpos_torch = torch.from_numpy(qpos).float().to(device)
    qvel_torch = torch.from_numpy(qvel).float().to(device)
    set_to_env(env, qpos_torch[0], qvel_torch[0], ctrl_torch[0])
    print(f"Environment created with {env.num_envs} environment(s)")

    # Rollout control sequence
    print(f"Rolling out {ctrl.shape[0]} control steps...")
    body_names = [env.sim.mj_model.body(i).name for i in range(env.sim.mj_model.nbody)]
    joint_names = [env.sim.mj_model.joint(i).name for i in range(env.sim.mj_model.njnt) if env.sim.mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
    joint_qpos_addr = [env.sim.mj_model.joint(i).qposadr for i in range(env.sim.mj_model.njnt) if env.sim.mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
    joint_qvel_addr = [env.sim.mj_model.joint(i).dofadr for i in range(env.sim.mj_model.njnt) if env.sim.mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_HINGE]
    
    motion_frames = []
    for i in range(ctrl.shape[0]):
        # Create tensordict with action
        action = ctrl_torch[i : i + 1]  # Shape: (1, nu)
        action = action.repeat(env.num_envs, 1)
        tensordict = TensorDict(
            {"action": action},
            batch_size=[env.num_envs],
            device=device,
        )

        # Step environment (handles decimation, viewer, command manager, etc.)
        # env.step(tensordict)

        if i % 1 == 0:
            set_to_env(env, qpos_torch[i], qvel_torch[i], ctrl_torch[i])
            
        # read xpos, xmat and cvel to get body link pos/quat and com lin/ang vel
        env.sim.forward()
        body_link_pos = wp.to_torch(env.sim.wp_data.xpos)
        body_link_quat = wp.to_torch(env.sim.wp_data.xquat)

        cvel = wp.to_torch(env.sim.wp_data.cvel)
        body_com_lin_vel = cvel[..., 3:]
        body_com_ang_vel = cvel[..., :3]
        joint_pos = wp.to_torch(env.sim.wp_data.qpos)[:, joint_qpos_addr]
        joint_vel = wp.to_torch(env.sim.wp_data.qvel)[:, joint_qvel_addr]
        motion_frames.append(TensorDict({
            "body_pos_w": body_link_pos.clone(),
            "body_quat_w": body_link_quat.clone(),
            "body_lin_vel_w": body_com_lin_vel.clone(),
            "body_ang_vel_w": body_com_ang_vel.clone(),
            "joint_pos": joint_pos.squeeze(-1),
            "joint_vel": joint_vel.squeeze(-1),
        }))
    motion_frames = torch.stack(motion_frames, dim=1)  # (num_envs, num_frames, ...)
    fps = int(1 / env.step_dt)
    print(f"Saving motion frames with shape: {motion_frames.shape} at {fps} FPS")
    save_path = "output"
    # save meta.json with fps and body/joint names
    # remove / prefix from body/joint names
    # robot/pelvis -> pelvis
    body_names = [name.split("/", 1)[-1] for name in body_names]
    joint_names = [name.split("/", 1)[-1] for name in joint_names]
    import json
    meta = {
        "fps": fps,
        "body_names": body_names,
        "joint_names": joint_names,
    }
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
    # save motion_frames as npz
    motion_frames_np = motion_frames[0].cpu().numpy()
    object_contact = np.zeros((motion_frames_np["body_pos_w"].shape[0], 1), dtype=np.bool_)
    object_contact[170:340] = True  # example contact frames
    # breakpoint()
    np.savez_compressed(
        os.path.join(save_path, "motion.npz"),
        **motion_frames_np,
        object_contact=object_contact,
    )

    print("Rollout complete!")


if __name__ == "__main__":
    main()
