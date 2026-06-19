# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from franka_isaaclab.assets.robots.doosan import DOOSAN_E0509_CFG

@configclass
class DoosanReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to doosan e0509
        self.scene.robot = DOOSAN_E0509_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override rewards to track link_6 (which is the Doosan end-effector/tool flange)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_6"]

        # override actions for 6-DOF Doosan E0509 (absolute actions with scaling and default offsets)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[1-6]"], scale=0.5, use_default_offset=True
        )
        
        # override command generator body to target link_6
        self.commands.ee_pose.body_name = "link_6"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class DoosanReachEnvCfg_PLAY(DoosanReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
