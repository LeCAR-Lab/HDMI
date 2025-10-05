import torch
import hydra
import numpy as np
import einops
import itertools
import os
import datetime
import re
from omegaconf import OmegaConf

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictSequential

from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.wandb import parse_checkpoint_path


@hydra.main(config_path="../cfg", config_name="play", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    from scripts.helpers import EpisodeStats, make_env_policy, ObsNorm, ObsOODDetector
    env, policy, vecnorm = make_env_policy(cfg)
    
    if cfg.export_policy:
        import time
        import copy
        
        # Load checkpoint to get wandb info and checkpoint number
        checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
        wandb_run_id = "unknown"
        checkpoint_num = "unknown"
        
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, weights_only=False)
            # Get wandb run ID from state_dict
            if "wandb" in state_dict and "id" in state_dict["wandb"]:
                wandb_run_id = state_dict["wandb"]["id"]
            
            # Extract checkpoint number from filename
            filename = os.path.basename(checkpoint_path)
            match = re.search(r'checkpoint_(\d+)', filename)
            if match:
                checkpoint_num = match.group(1)
            elif filename.endswith('_final.pt'):
                checkpoint_num = "final"
        
        fake_input = env.observation_spec[0].rand().cpu()
        fake_input["is_init"] = torch.tensor(1, dtype=bool)
        fake_input["context_adapt_hx"] = torch.zeros(128)
        fake_input = fake_input.unsqueeze(0)

        def test(m, x):
            start = time.perf_counter()
            for _ in range(1000):
                m(x)
            return (time.perf_counter() - start) / 1000
        
        FILE_PATH = os.path.dirname(__file__)
        
        deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
        obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
        ood_detector = ObsOODDetector(deploy_policy.in_keys, sigma=5.0)
        _policy = TensorDictSequential(obs_norm, ood_detector, deploy_policy).cpu()
        
        print(f"Inference time of policy: {test(_policy, fake_input)}")

        # Use new filename format with wandb_run_id and checkpoint_num
        os.makedirs(os.path.join(FILE_PATH, "exports", cfg.task.name), exist_ok=True)
        path = os.path.join(FILE_PATH, "exports", cfg.task.name, f"policy-{wandb_run_id}-{checkpoint_num}.pt")
        torch.save(_policy, path)

        meta = {}
        export_onnx(_policy, fake_input, path.replace(".pt", ".onnx"), meta)

        # export policy config
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        ## observation
        policy_config = dict()
        obs_cfg = dict()
        for k in deploy_policy.in_keys:
            obs_cfg[k] = dict_cfg["task"]["observation"][k]
        policy_config["observation"] = obs_cfg
        
        ## action
        policy_config["action_scale"] = dict_cfg["task"]["action"]["action_scaling"]

        ## joint names and stiffness/damping
        from active_adaptation.assets import get_asset_meta
        asset_meta = get_asset_meta(env.scene["robot"])
        policy_config["isaac_joint_names"] = asset_meta["joint_names_isaac"]
        joint_kp, joint_kd = {}, {}
        for actuator_name, actuator in asset_meta["actuators"].items():
            stiffness = actuator["stiffness"]
            if isinstance(stiffness, float):
                joint_name_expr = actuator["joint_names_expr"]
                if not isinstance(joint_name_expr, list):
                    joint_name_expr = [joint_name_expr]
                for joint_name in joint_name_expr:
                    joint_kp.update({joint_name: stiffness})
            elif isinstance(stiffness, dict):
                joint_kp.update(stiffness)
            else:
                raise ValueError(f"Unsupported stiffness type: {type(stiffness)}")

            damping = actuator["damping"]
            if isinstance(damping, float):
                joint_name_expr = actuator["joint_names_expr"]
                if not isinstance(joint_name_expr, list):
                    joint_name_expr = [joint_name_expr]
                for joint_name in joint_name_expr:
                    joint_kd.update({joint_name: damping})
            elif isinstance(damping, dict):
                joint_kd.update(damping)
            else:
                raise ValueError(f"Unsupported damping type: {type(damping)}")

        policy_config["joint_kp"] = joint_kp
        policy_config["joint_kd"] = joint_kd
        policy_config["default_joint_pos"] = asset_meta["init_state"]["joint_pos"]

        ## policy joint names
        from active_adaptation.envs.mdp.action import JointPosition
        action_manager: JointPosition = env.action_manager
        policy_config["policy_joint_names"] = action_manager.joint_names

        ## command
        command = env.command_manager
        cmd_key = "command" if "command" in policy_config["observation"] else "command_"
        command_obs = policy_config["observation"][cmd_key]
        if cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotTracking":
            from active_adaptation.envs.mdp.commands.hdmi.command import RobotTracking
            assert isinstance(command, RobotTracking)
            assert command.dataset.num_motions == 1

            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            motion_duration_second = command.dataset.lengths[0].item() * env.step_dt
            future_steps = command.future_steps.tolist()
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            root_body_name = command.root_body_name

            for obs_key in command_obs:
                command_obs[obs_key]["motion_duration_second"] = motion_duration_second
                command_obs[obs_key]["motion_path"] = cfg.task.command.data_path
                command_obs[obs_key]["future_steps"] = future_steps
                command_obs[obs_key]["body_names"] = tracking_keypoint_names
                command_obs[obs_key]["joint_names"] = tracking_joint_names
                command_obs[obs_key]["root_body_name"] = root_body_name
        elif cfg.task.command._target_ == "active_adaptation.envs.mdp.commands.hdmi.command.RobotObjectTracking":
            from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking
            assert isinstance(command, RobotObjectTracking)
            assert command.dataset.num_motions == 1

            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            motion_duration_second = command.dataset.lengths[0].item() * env.step_dt
            future_steps = command.future_steps.tolist()
            tracking_keypoint_names = command.tracking_keypoint_names
            tracking_joint_names = command.tracking_joint_names
            root_body_name = command.root_body_name

            # for motion observation
            for obs_key in command_obs:
                command_obs[obs_key]["motion_duration_second"] = motion_duration_second
                command_obs[obs_key]["motion_path"] = cfg.task.command.data_path
                command_obs[obs_key]["future_steps"] = future_steps
                command_obs[obs_key]["body_names"] = tracking_keypoint_names
                command_obs[obs_key]["joint_names"] = tracking_joint_names
                command_obs[obs_key]["root_body_name"] = root_body_name
            
            object_asset_name = cfg.task.command.object_asset_name
            object_body_name = cfg.task.command.object_body_name
            contact_target_pos_offset = np.array(cfg.task.command.contact_target_pos_offset).tolist()
            # for object observation in object obs
            object_obs = policy_config["observation"].get("object", None)
            if object_obs is not None:
                for obs_key in object_obs:
                    if obs_key == "ref_contact_pos_b":
                        object_obs[obs_key]["object_name"] = object_body_name
                        object_obs[obs_key]["contact_target_pos_offset"] = contact_target_pos_offset
                    else:
                        object_obs[obs_key]["object_name"] = object_asset_name
                    object_obs[obs_key]["root_body_name"] = root_body_name
            
        import yaml
        with open(path.replace(".pt", ".yaml"), "w") as f:
            yaml.dump(policy_config, f, sort_keys=False)
        print(f"Policy config saved to {path.replace('.pt', '.yaml')}")

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    td_ = env.reset()
    assert not env.base_env.training
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        torch.compiler.cudagraph_mark_step_begin()
        for i in itertools.count():
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            # td_.update(td["next"])
            episode_stats.add(td)

            if len(episode_stats) >= env.num_envs:
                print("Step", i)
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    print(k, torch.mean(v).item())
    
    env.close()


if __name__ == "__main__":
    main()