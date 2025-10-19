from __future__ import annotations

from pathlib import Path
from typing import Callable

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg


_ASSET_ROOT = Path(__file__).resolve().parent
_MJCF_OBJECTS_ROOT = (_ASSET_ROOT.parent / "assets" / "objects").resolve()

def read_assets_from_dir(asset_dir: Path) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for file_path in asset_dir.rglob("*"):
        if file_path.is_file():
            with open(file_path, "rb") as f:
                relative_path = file_path.relative_to(asset_dir).as_posix()
                assets[relative_path] = f.read()
    return assets


def _spec_loader(*relative_parts: str) -> Callable[[], mujoco.MjSpec]:
    xml_path = _MJCF_OBJECTS_ROOT.joinpath(*relative_parts)

    def _load() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        texture_dir = xml_path.parent / spec.texturedir
        mesh_dir = xml_path.parent / spec.meshdir
        spec.assets.update(read_assets_from_dir(texture_dir))
        spec.assets.update(read_assets_from_dir(mesh_dir))
        return spec

    return _load


class CustomEntity(Entity):
    """Entity with a single actuated joint and custom friction control."""

    def __init__(self, cfg: EntityCfg) -> None:
        super().__init__(cfg)
        self._device: str
        self._custom_friction: torch.Tensor
        self._custom_damping: torch.Tensor
        self.custom_torques: torch.Tensor

        # Add a motor actuator
        self.custom_joint_id: int = 0
        act = self.spec.add_actuator(
            name="custom_actuator",
            target=self.joint_names[self.custom_joint_id],
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            forcerange=(-100.0, 100.0),
        )
        act.gainprm[0] = 1.0

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("Entity not initialized yet.")
        return self._device

    def initialize(self, mj_model, model, data, device: str) -> None:  # noqa: D401
        super().initialize(mj_model, model, data, device)
        self._device = device


        if self.num_joints == 0:
            raise ValueError("CustomEntity requires at least one joint.")

        num_envs = data.nworld
        self._custom_friction = torch.zeros(num_envs, device=device)
        self._custom_damping = torch.zeros(num_envs, device=device)
        self.custom_torques = torch.zeros(num_envs, device=device)

    def write_data_to_sim(self) -> None:  # noqa: D401
        joint_vel = self.data.joint_vel[:, self.custom_joint_id]
        friction = -torch.sign(joint_vel) * (joint_vel.abs() > 0.01).float() * self._custom_friction
        damping = -joint_vel * self._custom_damping
        total = friction + damping

        self.custom_torques[:] = total
        self.data.write_ctrl(total.unsqueeze(-1), ctrl_ids=[self.custom_joint_id])

        super().write_data_to_sim()


DOOR_CFG = EntityCfg(class_type=CustomEntity, spec_fn=_spec_loader("door", "door.xml"))
BOX_CFG = EntityCfg(spec_fn=_spec_loader("box", "box_t1.xml"))
BOX_SMALL_CFG = EntityCfg(spec_fn=_spec_loader("box", "box.xml"))
SUITCASE_CFG = EntityCfg(spec_fn=_spec_loader("suitcase", "suitcase.xml"))
LARGEBOX_CFG = EntityCfg(spec_fn=_spec_loader("largebox", "largebox.xml"))
FOAM_CFG = EntityCfg(spec_fn=_spec_loader("foam", "foam.xml"))
STOOL_CFG = EntityCfg(spec_fn=_spec_loader("stool", "stool.xml"))
STOOL_LOW_CFG = EntityCfg(spec_fn=_spec_loader("stool", "stool-low.xml"))
STOOL_LOW2_CFG = EntityCfg(spec_fn=_spec_loader("stool", "stool_t1.xml"))
STAIR_CFG = EntityCfg(spec_fn=_spec_loader("stair", "stair.xml"))
SUPPORT0_CFG = EntityCfg(spec_fn=_spec_loader("support", "support0.xml"))
SUPPORT1_CFG = EntityCfg(spec_fn=_spec_loader("support", "support1.xml"))
STOOL_SUPPORT_CFG = EntityCfg(spec_fn=_spec_loader("stool", "stool_support.xml"))
WALL0_CFG = EntityCfg(spec_fn=_spec_loader("platform", "wall0.xml"))
PLATFORM0_CFG = EntityCfg(spec_fn=_spec_loader("platform", "platform0.xml"))
PLATFORM1_CFG = EntityCfg(spec_fn=_spec_loader("platform", "platform1.xml"))
BREAD_BOX_CFG = EntityCfg(spec_fn=_spec_loader("bread_box", "bread_box.xml"))
WOOD_BOARD_CFG = EntityCfg(spec_fn=_spec_loader("wood_board", "wood_board.xml"))
BALL_CFG = EntityCfg(spec_fn=_spec_loader("ball", "ball.xml"))
FOLDCHAIR_CFG = EntityCfg(class_type=CustomEntity, spec_fn=_spec_loader("foldchair", "foldchair.xml"))


OBJECTS_MJLAB = {
    "door": DOOR_CFG,
    "box": BOX_CFG,
    "box_small": BOX_SMALL_CFG,
    "suitcase": SUITCASE_CFG,
    "largebox": LARGEBOX_CFG,
    "foam": FOAM_CFG,
    "stool": STOOL_CFG,
    "stool_low": STOOL_LOW_CFG,
    "stool_low2": STOOL_LOW2_CFG,
    "stair": STAIR_CFG,
    "support0": SUPPORT0_CFG,
    "support1": SUPPORT1_CFG,
    "stool_support": STOOL_SUPPORT_CFG,
    "wall0": WALL0_CFG,
    "platform0": PLATFORM0_CFG,
    "platform1": PLATFORM1_CFG,
    "bread_box": BREAD_BOX_CFG,
    "wood_board": WOOD_BOARD_CFG,
    "ball": BALL_CFG,
    "foldchair": FOLDCHAIR_CFG,
}
