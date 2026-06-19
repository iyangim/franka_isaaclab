# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Doosan Robotics E0509 robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

DOOSAN_E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/iyangim/smart-shelf-robot/src/external/doosan-robot2/dsr_description2/usd/e0509.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.57,
            "joint_4": 0.0,
            "joint_5": 1.57,
            "joint_6": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "doosan_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_?[1-6]"],
            stiffness=800.0,
            damping=40.0,
            friction=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Doosan Robotics E0509 robot."""
