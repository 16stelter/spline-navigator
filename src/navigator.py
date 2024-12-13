#!/usr/bin/env python3

import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from scipy.interpolate import splprep, splev
from coverage_path_planning.msg import PointArr
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

        self.create_subscription(Odometry, f'{ns}/odom', self.odom_cb, 10)
        self.create_subscription(PointArr, f'{ns}/path', self.path_cb, 10) 
        self.cmd_vel_pub = self.create_publisher(Twist, f'{ns}/cmd_vel', 10)


    def odom_cb(self, msg):
        self.current_pose = msg

    def path_cb(self, msg):
        print("Received path")
        print(msg.data)
        x = np.array([self.current_pose.pose.pose.position.x])
        y = np.array([self.current_pose.pose.pose.position.y])
        for p in msg.data:
            x = np.append(x, p.x)
            y = np.append(y, p.y)
        points = np.vstack((x, y))

        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)
        print("Total distance: " + str(total_distance))
        num_samples = np.ceil(total_distance / 0.1).astype(int)
        print(len(x))

        tck, _ = splprep(points, k=min(3, len(x)-1), s=0)
        t = np.linspace(0, 1, num_samples)
        self.waypoints = np.array(splev(t, tck))
        plt.plot(x, y, 'o', label='Original Points')
        plt.plot(self.waypoints[0], self.waypoints[1], '-')
        plt.legend()
        plt.show()
        self.waypoints = self.waypoints.T
        print(self.waypoints)
        self.waypoint_idx = 0
   
    def navigate(self):
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.waypoint_idx <= len(self.waypoints)-1 and len(self.waypoints) > 1 and self.current_pose is not None:
                self.step()

    def step(self):
        waypoint = self.waypoints[self.waypoint_idx]
        posex = self.current_pose.pose.pose.position.x
        posey = self.current_pose.pose.pose.position.y

        dist = np.sqrt((waypoint[0]-posex)**2 + (waypoint[1]-posey)**2)

        if(dist < 0.1):
            self.waypoint_idx += 1
            print(str(self.waypoint_idx) + "/" + str(len(self.waypoints)))
            if(self.waypoint_idx == len(self.waypoints)):
                twist = Twist()
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                print("Reached goal")
            return
        
        twist = Twist()
        target_angle = np.arctan2(waypoint[1]-posey, waypoint[0]-posex)
        _, _, current_angle = self.euler_from_quaternion(self.current_pose.pose.pose.orientation.x, 
                                                         self.current_pose.pose.pose.orientation.y,
                                                         self.current_pose.pose.pose.orientation.z,
                                                         self.current_pose.pose.pose.orientation.w)
        angle_dist = target_angle - current_angle
        while angle_dist > np.pi:
            angle_dist -= 2 * np.pi
        while angle_dist < -np.pi:
            angle_dist += 2 * np.pi


        if abs(angle_dist) < 0.1:
            self.speed = min(self.max_speed, self.speed + self.accel)
        elif abs(angle_dist) > 0.2:
            self.speed = max(self.min_speed, self.speed - self.deccel)

        twist.linear.x = self.speed
        twist.angular.z = min(self.max_rotation, abs(angle_dist)) * np.sign(angle_dist)

        self.cmd_vel_pub.publish(twist)

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
    nav.navigate()
    nav.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


        