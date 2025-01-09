#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from spline_navigator.action import NavigatePath
from scipy.interpolate import splprep, splev
from rclpy.action import ActionServer, CancelResponse
import matplotlib.pyplot as plt

class Navigator(Node):
    def __init__(self):
        super().__init__("navigator")
        ns = self.get_namespace()
        if ns == "/":
            ns = ""

        self.max_speed = 2.0
        self.min_speed = 0.0
        self.accel = 0.02
        self.deccel = 0.05
        self.max_rotation = 3.0
        self.speed = 0.0

        self.waypoints = []
        self.current_pose = None
        self.waypoint_idx = 0
        self.last_waypoint_idx = -1

        self._action_server = ActionServer(
            self,
            NavigatePath,
            f"{ns}/navigate_path",
            self.execute_cb,
            cancel_callback=self.cancel_callback,
        )

        self.create_subscription(Odometry, f'{ns}/odom', self.odom_cb, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)


    def odom_cb(self, msg):
        self.current_pose = msg

    def execute_cb(self, goal_handle):
        self.get_logger().info("Received path")
        self.get_logger().info(f'{goal_handle.request.path}')
        # assuming we already stand in the first node, otherwise add
        # x = np.array([self.current_pose.pose.pose.position.x])
        # y = np.array([self.current_pose.pose.pose.position.y])
        x = np.array([])
        y = np.array([])
        for p in goal_handle.request.path:
            x = np.append(x, p.x)
            y = np.append(y, p.y)
        points = np.vstack((x, y))
        self.get_logger().info(f'Navigating region with {len(x)} waypoints.')

        if len(x) > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)
            total_distance = np.sum(distances)
            self.get_logger().info("Total distance: " + str(total_distance))
            
            self.waypoints = points.T

            self.waypoint_idx = 0
            success = self.navigate()
        else:
            self.get_logger().info("Skipping region with less than 2 waypoints.")
            success = True
        if success:
            self.get_logger().info("Goal succeeded")
            goal_handle.succeed()
        else:
            self.get_logger().info("Goal aborted")
            goal_handle.abort()
        result = NavigatePath.Result()
        result.success = success
        return result
   
    def cancel_callback(self, cancel_request):
        self.get_logger().warning('Cancelling goal...')
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().warning('Sent stop command.')
        return CancelResponse.ACCEPT

    def navigate(self):
        path_complete = False
        while rclpy.ok():
            if len(self.waypoints) > 1 and self.current_pose is not None:
                if self.waypoint_idx <= len(self.waypoints):
                    path_complete = self.step()
                else:
                    path_complete = True
            if path_complete:
                return True        

    def step(self):
        if self.last_waypoint_idx != self.waypoint_idx:
            self.get_logger().info("Progress: " + str(self.waypoint_idx + 1) + "/" + str(len(self.waypoints)))
            self.last_waypoint_idx = self.waypoint_idx
        if self.waypoint_idx >= len(self.waypoints):
            # Stop when all waypoints are reached
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Reached goal")
            return True

        # Current robot pose
        pose_x = self.current_pose.pose.pose.position.x
        pose_y = self.current_pose.pose.pose.position.y
        target_x = self.waypoints[self.waypoint_idx][0]
        target_y = self.waypoints[self.waypoint_idx][1]
       
        dist_to_target = np.sqrt((target_x - pose_x)**2 + (target_y - pose_y)**2)
        target_angle = np.arctan2(target_y - pose_y, target_x - pose_x)

        if dist_to_target < 0.1:
            self.waypoint_idx += 1
            return False

        _, _, current_angle = self.euler_from_quaternion(
            self.current_pose.pose.pose.orientation.x,
            self.current_pose.pose.pose.orientation.y,
            self.current_pose.pose.pose.orientation.z,
            self.current_pose.pose.pose.orientation.w
        )

        beta = target_angle - current_angle
        if beta > np.pi:
            beta -= 2*np.pi
        if beta < -np.pi:
            beta += 2*np.pi
        
        twist = Twist()
        if abs(beta) > 0.1:
            twist.linear.x = 0.0
        else:
            twist.linear.x = self.max_speed
        twist.angular.z = beta * 2
        self.cmd_vel_pub.publish(twist)
        return False

    def euler_from_quaternion(self, x, y ,z ,w):
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z


def main(args=None):
    rclpy.init(args=args)
    nav = Navigator()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(nav)

    try:
        executor.spin()
    finally:
        nav.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


        