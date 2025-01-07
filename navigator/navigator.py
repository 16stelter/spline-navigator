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
        aligned = False
        while rclpy.ok():
            rclpy.spin_once(self)
            if len(self.waypoints) > 1 and self.current_pose is not None:
                if self.waypoint_idx == 0 and not aligned:
                    aligned = self.align()
                elif self.waypoint_idx <= len(self.waypoints)-1 and aligned:
                    self.step()

    def align(self):
        waypoint = self.waypoints[self.waypoint_idx]
        posex = self.current_pose.pose.pose.position.x
        posey = self.current_pose.pose.pose.position.y
        
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

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = min(self.max_rotation, abs(angle_dist)) * np.sign(angle_dist)
        self.cmd_vel_pub.publish(twist)

        if abs(angle_dist) < 0.1:
            return True
        return False

    def step(self):
        print("Progress: " + str(self.waypoint_idx + 1) + "/" + str(len(self.waypoints)))
        if self.waypoint_idx + 1 >= len(self.waypoints):
            # Stop when all waypoints are reached
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            print("Reached goal")
            return

        # Current robot pose
        posex = self.current_pose.pose.pose.position.x
        posey = self.current_pose.pose.pose.position.y

        lookahead_dist = 0.5
        closest_idx, lookahead_point = self.find_lookahead_point((posex, posey), lookahead_dist)
        self.waypoint_idx = max(self.waypoint_idx, closest_idx)

        target_x, target_y = lookahead_point
        dist_to_target = np.sqrt((target_x - posex)**2 + (target_y - posey)**2)
        target_angle = np.arctan2(target_y - posey, target_x - posex)

        _, _, current_angle = self.euler_from_quaternion(
            self.current_pose.pose.pose.orientation.x,
            self.current_pose.pose.pose.orientation.y,
            self.current_pose.pose.pose.orientation.z,
            self.current_pose.pose.pose.orientation.w
        )
        angle_dist = target_angle - current_angle
        while angle_dist > np.pi:
            angle_dist -= 2 * np.pi
        while angle_dist < -np.pi:
            angle_dist += 2 * np.pi

        max_speed_factor = 1.0 - min(1.0, abs(angle_dist) / np.pi)  # Reduce speed for larger misalignment
        self.speed = min(self.max_speed * max_speed_factor, self.speed + self.accel)
        if abs(angle_dist) > 0.5:
            self.speed = max(self.min_speed, self.speed - self.deccel)  # Slow down sharply for large misalignment

        twist = Twist()
        twist.linear.x = self.speed
        twist.angular.z = angle_dist * 0.8  # Tunable proportional constant for smooth turning

        self.cmd_vel_pub.publish(twist)

    def find_lookahead_point(self, current_pos, lookahead_dist):
        posex, posey = current_pos
        closest_idx = self.waypoint_idx
        min_dist = float('inf')
        lookahead_point = None

        # Find the closest waypoint and lookahead point
        for i in range(self.waypoint_idx, len(self.waypoints)):
            wx, wy = self.waypoints[i]
            dist = np.sqrt((wx - posex)**2 + (wy - posey)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
            if dist > lookahead_dist and lookahead_point is None:
                lookahead_point = (wx, wy)
                break

        # Default to the last waypoint if no lookahead point is found
        if lookahead_point is None:
            lookahead_point = self.waypoints[-1]
            closest_idx = len(self.waypoints) - 1

        return closest_idx, lookahead_point



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


        