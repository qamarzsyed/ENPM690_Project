#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import time
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan

import numpy as np
import math
import cv2
from ray.rllib.policy.policy import Policy


LIN_VEL_STEP_SIZE = 1.25
ANG_VEL_STEP_SIZE = 0.09
model_path = "/home/qamar/Desktop/ENPM690_QSyed_FinalCode/TrainingFiles/BigL/checkpoint_000801/policies/default_policy"

class RLController(Node):

    def __init__(self):
        super().__init__('rl_controller')

        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        qos_profile = qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.linear_vel=0.0
        self.steer_angle=0.0

        self.kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 12))
        self.kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (3,3))
        self.policy = Policy.from_checkpoint(model_path)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)

    def action_map(self, output):
        joint_positions = Float64MultiArray()
        wheel_velocities = Float64MultiArray()
        if output is not None:
            if output == 4:  # Break
                self.linear_vel -= LIN_VEL_STEP_SIZE
                self.steer_angle = 0.8*(self.steer_angle)
            elif output == 3:  # Gas
                self.linear_vel += LIN_VEL_STEP_SIZE
                self.steer_angle = 0.8*(self.steer_angle)
            elif output == 2:  # Steer Left
                self.steer_angle += ANG_VEL_STEP_SIZE
            elif output == 1:  # Steer Right
                self.steer_angle -= ANG_VEL_STEP_SIZE
            # elif output == 0:  # Do Nothing
                  
            if self.steer_angle>0.5:
                self.steer_angle=0.5
            if self.steer_angle<-0.5:
                self.steer_angle=-0.5

            if self.linear_vel>17.5:
                self.linear_vel=17.5
            if self.linear_vel< 0.0:
                self.linear_vel= 0.0

            wheel_velocities.data = [self.linear_vel, self.linear_vel]
            joint_positions.data = [self.steer_angle, self.steer_angle]

            self.joint_position_pub.publish(joint_positions)
            self.wheel_velocities_pub.publish(wheel_velocities)

    def lidar_callback(self, lidar_msg: LaserScan):
        car_x, car_y = 72.0, 48.0
        track_image = np.zeros((144, 96, 3), dtype=np.uint8)
        a = 0
        for range_val in lidar_msg.ranges:
            a += 1
            rad_angle = (a/2)*(math.pi/180)
            if (math.isinf(range_val)):
                continue
            else:
                laser_x, laser_y = 11*range_val*math.sin(rad_angle), 11*range_val*math.cos(rad_angle)
                point_x, point_y = round(car_x+laser_x), round(car_y+laser_y)
                if (point_x < 144 and point_y < 96):
                    track_image[point_x, point_y, :] = [255, 255, 255]


        track_image = cv2.dilate(track_image, self.kernel1, iterations = 3)
        for i in range(track_image.shape[0]):
            point1_found = None
            point2_found = None
            for j in range(20):
                point1 = track_image[i, 48-j, :]
                point2 = track_image[i, 48+j, :]
                if (np.all([point1 == 255])):
                    point1_found = j
                if (np.all([point2 == 255])):
                    point2_found = j
                if (point1_found and point2_found):
                    break
            if (point1_found and point2_found):
                track_image[i, 48-point1_found:48+point2_found, :] = [255, 255, 255]
            # elif point1_found:
            #     track_image[i, 48-point1_found:48-point1_found+18, :] = [255, 255, 255]
            # elif point2_found:
            #     track_image[i, 48+point2_found-18:48+point2_found, :] = [255, 255, 255]
            
        track_image = cv2.morphologyEx(track_image, cv2.MORPH_CLOSE, self.kernel2)
        track_image = cv2.morphologyEx(track_image, cv2.MORPH_CLOSE, self.kernel2)
        track_image = cv2.morphologyEx(track_image, cv2.MORPH_CLOSE, self.kernel2)
        track_image[72-9:72+9, 48-8:48+8, :] = [255, 0, 0]

        track_image = cv2.flip(track_image, 0)
        track_image = cv2.flip(track_image, 1)
        track_image = track_image[0:96, :, :]
        cv2.imwrite('ROS_IMG.jpg', track_image)
        output = self.policy.compute_single_action(track_image, explore=False)
        self.action_map(output[0])

def main(args=None):
    rclpy.init(args=args)
    controller = RLController()
    
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()