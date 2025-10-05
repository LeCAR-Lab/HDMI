import os
from mjlab.entity import Entity

ASSET_PATH = os.path.dirname(__file__)

from dataclasses import asdict


def get_asset_meta(asset: Entity):
    # if not asset.is_initialized:
    #     raise RuntimeError("Articulation is not initialized. Please wait until `sim.reset` is called.")
    meta = {
        "init_state": asdict(asset.cfg.init_state),
        "body_names_isaac": asset.body_names,
        "joint_names_isaac": asset.joint_names,
        "actuators": {},
    }
    meta["default_joint_pos"] = asset.data.default_joint_pos[0].tolist()
    meta["stiffness"] = asset.data.default_joint_stiffness[0].tolist()
    meta["damping"] = asset.data.default_joint_damping[0].tolist()

    for i, actuator in enumerate(asset.cfg.articulation.actuators):
        meta["actuators"][f"actuator_{i}"] = asdict(actuator)
    return meta

