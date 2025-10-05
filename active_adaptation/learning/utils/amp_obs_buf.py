"""
This is a class for loading motion data for AMP.

It can receive a config in the form of a list of observations, and provides a method to generate batches of amp observations.
"""

import torch
import inspect

from active_adaptation.utils.motion import MotionDataset
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_mul,
    quat_conjugate,
    matrix_from_quat,
    yaw_quat,
    quat_apply_inverse
)
from active_adaptation.utils.math import batchify
from mjlab.third_party.isaaclab.isaaclab.utils.string import resolve_matching_names
from typing import Dict, Type

yaw_quat = batchify(yaw_quat)
quat_apply_inverse = batchify(quat_apply_inverse)


def log_tensor_members(obj, prefix=""):
    """
    Recursively logs all tensor members of an object and their shapes.

    Args:
        obj: The object to inspect
        prefix: Prefix for nested attributes (used in recursion)

    Returns:
        dict: Dictionary mapping attribute paths to their tensor shapes
    """
    tensor_info = {}

    # Handle direct list/tuple input
    if isinstance(obj, (list, tuple)):
        # First collect all tensor shapes
        shapes = []
        dtypes = set()
        devices = set()
        total_memory = 0

        for i, item in enumerate(obj):
            if isinstance(item, torch.Tensor):
                shapes.append(item.shape)
                dtypes.add(str(item.dtype))
                devices.add(str(item.device))
                total_memory += (
                    item.element_size() * item.nelement() / (1024 * 1024)
                )  # Size in MB
            elif hasattr(item, "__dict__"):
                nested_info = log_tensor_members(item, f"{prefix}[{i}]")
                tensor_info.update(nested_info)

        # If we found any tensors in the list, consolidate their info
        if shapes:
            # Verify all shapes are the same
            if all(s == shapes[0] for s in shapes):
                tensor_info[f"{prefix}"] = {
                    "shape": f"List[{len(shapes)}] of tensors with shape {shapes[0]}",
                    "dtype": list(dtypes)[0] if len(dtypes) == 1 else list(dtypes),
                    "device": list(devices)[0] if len(devices) == 1 else list(devices),
                    "memory_mb": total_memory,
                    "num_items": len(shapes),
                }
            else:
                tensor_info[f"{prefix}"] = {
                    "shape": f"List[{len(shapes)}] of tensors with varying shapes {shapes}",
                    "dtype": list(dtypes)[0] if len(dtypes) == 1 else list(dtypes),
                    "device": list(devices)[0] if len(devices) == 1 else list(devices),
                    "memory_mb": total_memory,
                    "num_items": len(shapes),
                }
        return tensor_info

    for attr_name, attr_value in obj.__dict__.items():
        full_name = f"{prefix}.{attr_name}" if prefix else attr_name

        # Check if it's a tensor
        if isinstance(attr_value, torch.Tensor):
            tensor_info[full_name] = {
                "shape": tuple(attr_value.shape),
                "dtype": str(attr_value.dtype),
                "device": str(attr_value.device),
                "memory_mb": attr_value.element_size()
                * attr_value.nelement()
                / (1024 * 1024),  # Size in MB
            }

        # Check if it's a list/tuple of tensors
        elif isinstance(attr_value, (list, tuple)):
            for i, item in enumerate(attr_value):
                if isinstance(item, torch.Tensor):
                    tensor_info[f"{full_name}[{i}]"] = {
                        "shape": tuple(item.shape),
                        "dtype": str(item.dtype),
                        "device": str(item.device),
                        "memory_mb": item.element_size()
                        * item.nelement()
                        / (1024 * 1024),  # Size in MB
                    }

        # Check if it's a dict of tensors
        elif isinstance(attr_value, dict):
            for key, value in attr_value.items():
                if isinstance(value, torch.Tensor):
                    tensor_info[f"{full_name}[{key}]"] = {
                        "shape": tuple(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                        "memory_mb": value.element_size()
                        * value.nelement()
                        / (1024 * 1024),  # Size in MB
                    }

    return tensor_info

class _RegistryMixin:
    def __init_subclass__(cls):
        if not hasattr(cls, 'registry'):
            cls.registry: Dict[str, Type[AMPObservation]] = {}

        cls_name = cls.__name__
        try:
            cls._file = inspect.getfile(cls)
            cls._line = inspect.getsourcelines(cls)[1]
        except:
            cls._file = "unknown"
            cls._line = "unknown"
        
        if cls_name.startswith("_"):
            return
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls    
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")

class AMPObservation(_RegistryMixin):
    def __init__(self, buffer: "AMPObsBuffer", **kwargs):
        self.buffer = buffer

    def compute(self):
        """Compute the AMP observation."""
        raise NotImplementedError("This method should be implemented by subclasses.")

class AMPObsBuffer:
    def __init__(self, motion_lib: MotionDataset, obs_cfg: Dict):
        self.device = motion_lib.device

        self.body_link_pos_w = motion_lib.data.body_pos_w
        self.body_link_quat_w = motion_lib.data.body_quat_w
        self.body_link_lin_vel_w = motion_lib.data.body_lin_vel_w
        self.body_link_ang_vel_w = motion_lib.data.body_ang_vel_w
        self.joint_pos = motion_lib.data.joint_pos
        self.joint_vel = motion_lib.data.joint_vel

        self.body_names = motion_lib.body_names
        self.joint_names = motion_lib.joint_names
        self.motion_num_frames = motion_lib.lengths
        self.total_num_frames = motion_lib.data.shape[0]
        self.starts = motion_lib.starts
        self.ends = motion_lib.ends

        amp_obs_list = []
        self.obs_metadata = []  # Store metadata for each observation type
        
        print("AMP ObsBuffer: Computing AMP observations:")
        for obs_key, obs_params in obs_cfg.items():
            obs_class = AMPObservation.registry[obs_key]
            obs_instance = obs_class(buffer=self, **obs_params)
            amp_obs = obs_instance.compute()
            amp_obs_list.append(amp_obs)
            print(f"\t{obs_key}: {amp_obs.shape}")
            
            # Collect metadata
            metadata = {
                "obs_type": str(obs_key),
                "obs_dim": amp_obs.shape[-1],
            }
            
            # Add specific metadata for different observation types
            if hasattr(obs_instance, 'joint_names'):
                metadata["joint_names"] = obs_instance.joint_names
            if hasattr(obs_instance, 'body_names'):
                metadata["body_names"] = obs_instance.body_names
            if hasattr(obs_instance, 'history_steps'):
                metadata["history_steps"] = list(obs_instance.history_steps)
                
            self.obs_metadata.append(metadata)
        
        self.amp_obs_buf = torch.concat(amp_obs_list, dim=-1)
        # shape: (total_num_frames, obs_dim)

    def sample(self, num_samples):
        sample_indices = torch.randint(0, self.total_num_frames, (num_samples,))
        amp_obs = self.amp_obs_buf[sample_indices]  # (num_samples, obs_dim)
        return amp_obs

    def export(self, output_dir: str):
        """
        Export the AMP observation buffer and metadata to files.
        
        Args:
            output_dir: Directory to save files
            filename_prefix: Prefix for output files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tensor
        tensor_path = os.path.join(output_dir, f"tensor.pt")
        torch.save(self.amp_obs_buf.cpu(), tensor_path)
        
        # Prepare metadata
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"metadata.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(self.obs_metadata, f, indent=2)
        
        print(f"AMP observation buffer exported to:")
        print(f"  Tensor: {tensor_path}")
        print(f"  Metadata: {metadata_path}")
        
        return tensor_path, metadata_path


def _get_history_obs(obs: torch.Tensor, history_steps: list[int]) -> torch.Tensor:
    """
    Get history observations for a given observation tensor.

    Args:
        obs (torch.Tensor): The observation tensor of shape (num_frames, obs_dim).
        history_steps (list[int]): List of history steps to include.

    Returns:
        torch.Tensor: History observations of shape (num_frames, num_history_steps * obs_dim).
    """
    num_frames = obs.shape[0]
    max_history = max(history_steps)

    # Pad with first value repeated for history
    first_frame = obs[0:1].expand(max_history, *obs.shape[1:])  # (max_history, obs_dim)
    padded_obs = torch.cat([first_frame, obs], dim=0)

    current_indices = torch.arange(num_frames, device=obs.device) + max_history
    history_offsets = torch.tensor(history_steps, device=obs.device).view(1, -1)
    obs_indices = current_indices.unsqueeze(1) - history_offsets.unsqueeze(0)  # (num_frames, num_history_steps)
    
    return padded_obs[obs_indices].reshape(num_frames, -1)  # Flatten to (num_frames, num_history_steps * obs_dim)

class joint_pos_history_amp(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", joint_names: str = ".*", history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        joint_names = resolve_matching_names(joint_names, buffer.joint_names)[1]
        self.joint_names = list(sorted(joint_names))
        self.joint_ids = [buffer.joint_names.index(name) for name in self.joint_names]

    def compute(self):
        # Get joint positions for the selected joints
        joint_pos = self.buffer.joint_pos[:, self.joint_ids]  # (num_frames, num_joints)
        return _get_history_obs(joint_pos, self.history_steps)  # (num_frames, num_history_steps * num_joints)

class joint_vel_history_amp(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", joint_names: str = ".*", history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        joint_names = resolve_matching_names(joint_names, buffer.joint_names)[1]
        self.joint_names = list(sorted(joint_names))
        self.joint_ids = [buffer.joint_names.index(name) for name in self.joint_names]

    def compute(self):
        # Get joint positions for the selected joints
        joint_vel = self.buffer.joint_vel[:, self.joint_ids]  # (num_frames, num_joints)
        return _get_history_obs(joint_vel, self.history_steps)  # (num_frames, num_history_steps * num_joints)

class body_pos_b_history(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", body_names: list[str], history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        body_names = resolve_matching_names(body_names, buffer.body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [buffer.body_names.index(name) for name in self.body_names]
        self.root_body_idx = buffer.body_names.index("pelvis")
        
    def compute(self):
        root_link_pos_w = self.buffer.body_link_pos_w[:, self.root_body_idx]  # (num_frames, 3)
        root_link_quat_w = self.buffer.body_link_quat_w[:, self.root_body_idx]  # (num_frames, 4)

        # Zero out Z component and use yaw-only quaternion
        root_link_pos_w_flat = root_link_pos_w.clone()
        root_link_pos_w_flat[..., 2] = 0.0
        root_link_quat_w_yaw = yaw_quat(root_link_quat_w)
        
        # Get body positions
        body_link_pos_w = self.buffer.body_link_pos_w[:, self.body_indices]  # (num_frames, num_bodies, 3)
        
        # Transform to body frame
        body_pos_b = quat_apply_inverse(
            root_link_quat_w_yaw.unsqueeze(1),
            body_link_pos_w - root_link_pos_w_flat.unsqueeze(1)
        )
        
        return _get_history_obs(body_pos_b, self.history_steps)  # (num_frames, num_history_steps * num_bodies * 3)
    
class body_lin_vel_b_history(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", body_names: list[str], history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        body_names = resolve_matching_names(body_names, buffer.body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [buffer.body_names.index(name) for name in self.body_names]
        self.root_body_idx = buffer.body_names.index("pelvis")
        
    def compute(self):
        from active_adaptation.utils.math import yaw_quat
        
        # Get root orientation (assuming first body is root)
        root_link_quat_w = self.buffer.body_link_quat_w[:, self.root_body_idx]  # (num_frames, 4)
        root_link_quat_w_yaw = yaw_quat(root_link_quat_w)
        
        # Get body linear velocities
        body_link_lin_vel_w = self.buffer.body_link_lin_vel_w[:, self.body_indices]  # (num_frames, num_bodies, 3)
        
        # Transform to body frame
        body_lin_vel_b = quat_apply_inverse(
            root_link_quat_w_yaw.unsqueeze(1),
            body_link_lin_vel_w
        )
        return _get_history_obs(body_lin_vel_b, self.history_steps)

class body_ori_b_history(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", body_names: list[str], history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        body_names = resolve_matching_names(body_names, buffer.body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [buffer.body_names.index(name) for name in self.body_names]
        self.root_body_idx = buffer.body_names.index("pelvis")

    def compute(self):
        root_link_quat_w = self.buffer.body_link_quat_w[:, self.root_body_idx]  # (num_frames, 4)
        root_link_quat_w_yaw = yaw_quat(root_link_quat_w)
        
        # Get body orientations
        body_link_quat_w = self.buffer.body_link_quat_w[:, self.body_indices]  # (num_frames, num_bodies, 4)
        
        # Transform to body frame
        body_quat_b = quat_mul(
            quat_conjugate(root_link_quat_w_yaw).unsqueeze(1).expand_as(body_link_quat_w),
            body_link_quat_w,
        )
        
        # Convert to rotation matrix and extract first two columns
        body_ori_b = matrix_from_quat(body_quat_b)  # (num_frames, num_bodies, 3, 3)
        body_ori_b_reduced = body_ori_b[:, :, :2, :3]  # (num_frames, num_bodies, 2, 3)
        return _get_history_obs(body_ori_b_reduced, self.history_steps)

class body_ang_vel_b_history(AMPObservation):
    def __init__(self, buffer: "AMPObsBuffer", body_names: list[str], history_steps: list[int] = [0]):
        super().__init__(buffer)
        self.history_steps = history_steps
        body_names = resolve_matching_names(body_names, buffer.body_names)[1]
        self.body_names = list(sorted(body_names))
        self.body_indices = [buffer.body_names.index(name) for name in self.body_names]
        self.root_body_idx = buffer.body_names.index("pelvis")

    def compute(self):
        root_link_quat_w = self.buffer.body_link_quat_w[:, self.root_body_idx]  # (num_frames, 4)
        root_link_quat_w_yaw = yaw_quat(root_link_quat_w)
        
        # Get body angular velocities
        body_link_ang_vel_w = self.buffer.body_link_ang_vel_w[:, self.body_indices]  # (num_frames, num_bodies, 3)
        
        # Transform to body frame
        body_ang_vel_b = quat_apply_inverse(
            root_link_quat_w_yaw.unsqueeze(1),
            body_link_ang_vel_w
        )
        
        return _get_history_obs(body_ang_vel_b, self.history_steps)  # (num_frames, num_history_steps * num_bodies * 3)