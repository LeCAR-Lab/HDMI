import torch

import omni.isaac.core.utils.prims as prim_utils
from pxr import PhysxSchema

def attach_payload(parent_path):
    import omni.physx.scripts.utils as script_utils
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.core import objects
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.18, 0.16, 0.12]),
        mass=0.0001,
        translation=torch.tensor([0.0, 0.0, 0.1]),
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.0)


import carb
import omni
import weakref
from collections import defaultdict


class IsaacCameraControl:
    def __init__(self, env):
        self.env = env
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        self.key_pressed = defaultdict(lambda: False)
        self.focus = False
        self.lookat_env_i = self.env.num_envs // 2
        self.distance = 2.0
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "F" and not self.focus:
                self.focus = not self.focus
            elif event.input.name == "TAB":
                self.lookat_env_i = (self.lookat_env_i + 1) % self.env.num_envs
    
    def update(self):
        if self.focus:
            self.env.sim.set_camera_view(
                eye=self.robot.data.root_link_pos_w[self.lookat_env_i].cpu() + torch.ones(3) * self.distance,
                target=self.robot.data.root_link_pos_w[self.lookat_env_i].cpu(),
            )
