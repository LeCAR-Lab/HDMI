# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    yaw_quat,
    wrap_to_pi,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_conjugate,
    axis_angle_from_quat,
    create_rotation_matrix_from_view,
    convert_camera_frame_orientation_convention
)

from .helpers import batchify

@torch.jit.script
def quat_rotate(quat: torch.Tensor, vec: torch.Tensor):
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[..., 0:1] * t + xyz.cross(t, dim=-1))


@torch.jit.script
def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor):
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[..., 0:1] * t + xyz.cross(t, dim=-1))


def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp(1e-6)


def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis


def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor):
    """
    Rotate a vector by a yaw angle (in radians).
    """
    yaw_cos = torch.cos(yaw)
    yaw_sin = torch.sin(yaw)
    vec = vec.expand(*yaw.shape, 3)
    return torch.stack(
        [
            yaw_cos * vec[..., 0] - yaw_sin * vec[..., 1],
            yaw_sin * vec[..., 0] + yaw_cos * vec[..., 1],
            vec[..., 2],
        ],
        dim=-1,
    )


def quat_from_yaw(yaw: torch.Tensor):
    return torch.cat(
        [
            torch.cos(yaw / 2).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.sin(yaw / 2).unsqueeze(-1),
        ],
        dim=-1,
    )


def euler_from_quat(quat: torch.Tensor):
    w, x, y, z = quat.unbind(-1)
    # Convert quaternion to roll, pitch, yaw Euler angles
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sin_pitch) >= 1,
        torch.full_like(sin_pitch, torch.pi / 2.0) * torch.sign(sin_pitch),
        torch.asin(sin_pitch)
    )

    sin_yaw = 2.0 * (w * z + x * y) 
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_from_view(eyes: torch.Tensor, lookat: torch.Tensor):
    matrix = create_rotation_matrix_from_view(eyes, lookat, up_axis="Z", device=eyes.device)
    quat = quat_from_matrix(matrix)
    quat = convert_camera_frame_orientation_convention(quat, "opengl", "world")
    return quat


class MultiUniform(D.Distribution):
    """
    A distribution over the union of multiple disjoint intervals.
    """
    def __init__(self, ranges: torch.Tensor):
        batch_shape = ranges.shape[:-2]
        if not ranges[..., 0].le(ranges[..., 1]).all():
            raise ValueError("Ranges must be non-empty and ordered.")
        super().__init__(batch_shape, validate_args=False)
        self.ranges = ranges
        self.ranges_len = ranges.diff(dim=-1).squeeze(1)
        self.total_len = self.ranges_len.sum(-1)
        self.starts = torch.zeros_like(ranges[..., 0])
        self.starts[..., 1:] = self.ranges_len.cumsum(-1)[..., :-1]

    def sample(self, sample_shape: torch.Size = ()) -> torch.Tensor:
        sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.batch_shape
        uniform = torch.rand(shape, device=self.ranges.device) * self.total_len
        i = torch.searchsorted(self.starts, uniform) - 1
        return self.ranges[i, 0] + uniform - self.starts[i]



class EMA:
    """
    Exponential Moving Average.
    
    Args:
        x: The tensor to compute the EMA of.
        gammas: The decay rates. Can be a single float or a list of floats.
    
    Example:
        >>> ema = EMA(x, gammas=[0.9, 0.99])
        >>> ema.update(x)
        >>> ema.ema
    """
    def __init__(self, x: torch.Tensor, gammas):
        self.gammas = torch.tensor(gammas, device=x.device)
        shape = (x.shape[0], len(self.gammas), *x.shape[1:])
        self.sum = torch.zeros(shape, device=x.device)
        shape = (x.shape[0], len(self.gammas), 1)
        self.cnt = torch.zeros(shape, device=x.device)

    def reset(self, env_ids: torch.Tensor):
        self.sum[env_ids] = 0.0
        self.cnt[env_ids] = 0.0
        
    def update(self, x: torch.Tensor):
        self.sum.mul_(self.gammas.unsqueeze(-1)).add_(x.unsqueeze(1))
        self.cnt.mul_(self.gammas.unsqueeze(-1)).add_(1.0)
        self.ema = self.sum / self.cnt
        return self.ema

