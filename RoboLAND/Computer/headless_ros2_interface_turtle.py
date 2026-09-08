#!/usr/bin/env python3
"""Lightweight replica of ControlNode_Turtle without Kivy dependencies.

This keeps the subset of behaviour that terrain_data_collector.py needs:
  * publish_gui_information
  * telemetry subscriptions and buffers
  * calibrate/update_force_data/bookkeeping helpers
"""

from __future__ import annotations

import time
from typing import List

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ControlNode_Turtle(Node):
    def __init__(self) -> None:
        super().__init__("control_node")
        self.id = "turtle"
        self.publisher_ = self.create_publisher(Float64MultiArray, "/Gui_information", 10)

        self.create_subscription(Float64MultiArray, "/robot_state", self.turtle_status, 10)
        self.create_subscription(Pose, "/optitrack_body", self.OptitrackState, 10)
        self.create_subscription(Pose, "/optitrack_left_flipper", self.LeftFlipperState, 10)
        self.create_subscription(Pose, "/optitrack_right_flipper", self.RightFlipperState, 10)

        self.tab_control: List[str] = ["turtle"]

        # latest telemetry samples
        self.leftadduction_pos = 0.0
        self.leftsweeping_pos = 0.0
        self.rightadduction_pos = 0.0
        self.rightsweeping_pos = 0.0
        self.leftadduction_curr = 0.0
        self.leftsweeping_curr = 0.0
        self.rightadduction_curr = 0.0
        self.rightsweeping_curr = 0.0
        self.OptitrackPosition_x = 0.0
        self.OptitrackPosition_y = 0.0
        self.OptitrackPosition_z = 0.0
        self.OptitrackOrientation_x = 0.0
        self.OptitrackOrientation_y = 0.0
        self.OptitrackOrientation_z = 0.0
        self.OptitrackOrientation_w = 0.0
        self.LeftFlipperPosition_x = 0.0
        self.LeftFlipperPosition_y = 0.0
        self.LeftFlipperPosition_z = 0.0
        self.LeftFlipperOrientation_x = 0.0
        self.LeftFlipperOrientation_y = 0.0
        self.LeftFlipperOrientation_z = 0.0
        self.LeftFlipperOrientation_w = 0.0
        self.RightFlipperPosition_x = 0.0
        self.RightFlipperPosition_y = 0.0
        self.RightFlipperPosition_z = 0.0
        self.RightFlipperOrientation_x = 0.0
        self.RightFlipperOrientation_y = 0.0
        self.RightFlipperOrientation_z = 0.0
        self.RightFlipperOrientation_w = 0.0
        self.turtle_state = 0.0

        # Rolling buffers used for CSV export.
        self.time_list: List[float] = []
        self.leftadduction_pos_array: List[float] = []
        self.leftsweeping_pos_array: List[float] = []
        self.rightadduction_pos_array: List[float] = []
        self.rightsweeping_pos_array: List[float] = []
        self.leftadduction_curr_array: List[float] = []
        self.leftsweeping_curr_array: List[float] = []
        self.rightadduction_curr_array: List[float] = []
        self.rightsweeping_curr_array: List[float] = []
        self.OptitrackPosition_x_list: List[float] = []
        self.OptitrackPosition_y_list: List[float] = []
        self.OptitrackPosition_z_list: List[float] = []
        self.OptitrackOrientation_x_list: List[float] = []
        self.OptitrackOrientation_y_list: List[float] = []
        self.OptitrackOrientation_z_list: List[float] = []
        self.OptitrackOrientation_w_list: List[float] = []
        self.LeftFlipperPosition_x_list: List[float] = []
        self.LeftFlipperPosition_y_list: List[float] = []
        self.LeftFlipperPosition_z_list: List[float] = []
        self.LeftFlipperOrientation_x_list: List[float] = []
        self.LeftFlipperOrientation_y_list: List[float] = []
        self.LeftFlipperOrientation_z_list: List[float] = []
        self.LeftFlipperOrientation_w_list: List[float] = []
        self.RightFlipperPosition_x_list: List[float] = []
        self.RightFlipperPosition_y_list: List[float] = []
        self.RightFlipperPosition_z_list: List[float] = []
        self.RightFlipperOrientation_x_list: List[float] = []
        self.RightFlipperOrientation_y_list: List[float] = []
        self.RightFlipperOrientation_z_list: List[float] = []
        self.RightFlipperOrientation_w_list: List[float] = []
        self.turtle_state_list: List[float] = []

        self.start_time = time.time()

    # --- GUI-oriented helpers kept as no-ops for API compatibility ---
    def get_fp(self):
        return None

    def get_pp(self):
        return None

    def get_speed_p(self):
        return None

    # --- Control helpers used by terrain_data_collector.py ---
    def publish_gui_information(self, msg: Float64MultiArray) -> None:
        self.publisher_.publish(msg)

    def calibrate(self, *args, **kwargs) -> None:
        self.start_time = time.time()
        self.time_list.clear()
        self.leftadduction_pos_array.clear()
        self.leftsweeping_pos_array.clear()
        self.rightadduction_pos_array.clear()
        self.rightsweeping_pos_array.clear()
        self.leftadduction_curr_array.clear()
        self.leftsweeping_curr_array.clear()
        self.rightadduction_curr_array.clear()
        self.rightsweeping_curr_array.clear()
        self.OptitrackPosition_x_list.clear()
        self.OptitrackPosition_y_list.clear()
        self.OptitrackPosition_z_list.clear()
        self.OptitrackOrientation_x_list.clear()
        self.OptitrackOrientation_y_list.clear()
        self.OptitrackOrientation_z_list.clear()
        self.OptitrackOrientation_w_list.clear()
        self.LeftFlipperPosition_x_list.clear()
        self.LeftFlipperPosition_y_list.clear()
        self.LeftFlipperPosition_z_list.clear()
        self.LeftFlipperOrientation_x_list.clear()
        self.LeftFlipperOrientation_y_list.clear()
        self.LeftFlipperOrientation_z_list.clear()
        self.LeftFlipperOrientation_w_list.clear()
        self.RightFlipperPosition_x_list.clear()
        self.RightFlipperPosition_y_list.clear()
        self.RightFlipperPosition_z_list.clear()
        self.RightFlipperOrientation_x_list.clear()
        self.RightFlipperOrientation_y_list.clear()
        self.RightFlipperOrientation_z_list.clear()
        self.RightFlipperOrientation_w_list.clear()
        self.turtle_state_list.clear()

    def update_force_data(self, updateplotflag: bool) -> None:
        if not updateplotflag:
            return
        current_time = time.time() - self.start_time
        self.time_list.append(current_time)
        self.leftadduction_pos_array.append(self.leftadduction_pos)
        self.leftsweeping_pos_array.append(self.leftsweeping_pos)
        self.rightadduction_pos_array.append(self.rightadduction_pos)
        self.rightsweeping_pos_array.append(self.rightsweeping_pos)
        self.leftadduction_curr_array.append(self.leftadduction_curr)
        self.leftsweeping_curr_array.append(self.leftsweeping_curr)
        self.rightadduction_curr_array.append(self.rightadduction_curr)
        self.rightsweeping_curr_array.append(self.rightsweeping_curr)
        self.OptitrackPosition_x_list.append(self.OptitrackPosition_x)
        self.OptitrackPosition_y_list.append(self.OptitrackPosition_y)
        self.OptitrackPosition_z_list.append(self.OptitrackPosition_z)
        self.OptitrackOrientation_x_list.append(self.OptitrackOrientation_x)
        self.OptitrackOrientation_y_list.append(self.OptitrackOrientation_y)
        self.OptitrackOrientation_z_list.append(self.OptitrackOrientation_z)
        self.OptitrackOrientation_w_list.append(self.OptitrackOrientation_w)
        self.LeftFlipperPosition_x_list.append(self.LeftFlipperPosition_x)
        self.LeftFlipperPosition_y_list.append(self.LeftFlipperPosition_y)
        self.LeftFlipperPosition_z_list.append(self.LeftFlipperPosition_z)
        self.LeftFlipperOrientation_x_list.append(self.LeftFlipperOrientation_x)
        self.LeftFlipperOrientation_y_list.append(self.LeftFlipperOrientation_y)
        self.LeftFlipperOrientation_z_list.append(self.LeftFlipperOrientation_z)
        self.LeftFlipperOrientation_w_list.append(self.LeftFlipperOrientation_w)
        self.RightFlipperPosition_x_list.append(self.RightFlipperPosition_x)
        self.RightFlipperPosition_y_list.append(self.RightFlipperPosition_y)
        self.RightFlipperPosition_z_list.append(self.RightFlipperPosition_z)
        self.RightFlipperOrientation_x_list.append(self.RightFlipperOrientation_x)
        self.RightFlipperOrientation_y_list.append(self.RightFlipperOrientation_y)
        self.RightFlipperOrientation_z_list.append(self.RightFlipperOrientation_z)
        self.RightFlipperOrientation_w_list.append(self.RightFlipperOrientation_w)
        self.turtle_state_list.append(self.turtle_state)

    # --- ROS 2 subscription callbacks ---
    def turtle_status(self, msg: Float64MultiArray) -> None:
        self.turtle_state = msg.data[0]
        self.leftadduction_pos = msg.data[1]
        self.leftsweeping_pos = msg.data[2]
        self.rightadduction_pos = msg.data[3]
        self.rightsweeping_pos = msg.data[4]
        self.leftadduction_curr = msg.data[5]
        self.leftsweeping_curr = msg.data[6]
        self.rightadduction_curr = msg.data[7]
        self.rightsweeping_curr = msg.data[8]

    def OptitrackState(self, msg: Pose) -> None:
        self.OptitrackPosition_x = msg.position.x
        self.OptitrackPosition_y = msg.position.y
        self.OptitrackPosition_z = msg.position.z
        self.OptitrackOrientation_x = msg.orientation.x
        self.OptitrackOrientation_y = msg.orientation.y
        self.OptitrackOrientation_z = msg.orientation.z
        self.OptitrackOrientation_w = msg.orientation.w

    def LeftFlipperState(self, msg: Pose) -> None:
        self.LeftFlipperPosition_x = msg.position.x
        self.LeftFlipperPosition_y = msg.position.y
        self.LeftFlipperPosition_z = msg.position.z
        self.LeftFlipperOrientation_x = msg.orientation.x
        self.LeftFlipperOrientation_y = msg.orientation.y
        self.LeftFlipperOrientation_z = msg.orientation.z
        self.LeftFlipperOrientation_w = msg.orientation.w

    def RightFlipperState(self, msg: Pose) -> None:
        self.RightFlipperPosition_x = msg.position.x
        self.RightFlipperPosition_y = msg.position.y
        self.RightFlipperPosition_z = msg.position.z
        self.RightFlipperOrientation_x = msg.orientation.x
        self.RightFlipperOrientation_y = msg.orientation.y
        self.RightFlipperOrientation_z = msg.orientation.z
        self.RightFlipperOrientation_w = msg.orientation.w


__all__ = ["ControlNode_Turtle"]
