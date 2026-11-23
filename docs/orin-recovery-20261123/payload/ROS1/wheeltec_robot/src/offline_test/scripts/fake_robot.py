#!/usr/bin/python3
"""Publish simulated /scan, /odom and odom_combined->base_footprint TF.

The laser scan is ray-cast from the loaded occupancy map, so AMCL sees a
self-consistent world. /cmd_vel is integrated like a differential-drive base,
which lets move_base/TEB drive the simulated robot without any hardware.
"""

import math
import os

import numpy as np
import rospy
import tf
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def load_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    tokens = []
    pos = 0
    while len(tokens) < 4:
        nl = data.index(b"\n", pos)
        line = data[pos:nl].strip()
        pos = nl + 1
        if not line or line.startswith(b"#"):
            continue
        tokens.extend(line.split())
    width, height = int(tokens[1]), int(tokens[2])
    img = np.frombuffer(data[pos:pos + width * height], dtype=np.uint8)
    return img.reshape(height, width), width, height


class FakeRobot:
    def __init__(self):
        self.map_yaml = rospy.get_param("~map_yaml")
        with open(self.map_yaml, "r") as f:
            meta = yaml.safe_load(f)
        image = meta["image"]
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(self.map_yaml), image)
        self.grid, self.width, self.height = load_pgm(image)
        self.resolution = float(meta["resolution"])
        self.origin_x = float(meta["origin"][0])
        self.origin_y = float(meta["origin"][1])
        self.occupied = self.grid < 128

        self.x = float(rospy.get_param("~init_x", 0.0))
        self.y = float(rospy.get_param("~init_y", 0.0))
        self.yaw = float(rospy.get_param("~init_yaw", 0.0))

        self.range_max = 10.0
        self.beam_count = 360
        self.twist = Twist()
        self.last_cmd_time = rospy.Time(0)

        self.scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=1)
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        rospy.Subscriber("/cmd_vel", Twist, self.cmd_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / 30.0), self.odom_timer)
        rospy.Timer(rospy.Duration(0.1), self.scan_timer)
        rospy.loginfo("fake_robot ready: pose=(%.2f, %.2f, %.2f), map=%s",
                      self.x, self.y, self.yaw, image)
        rospy.loginfo("send a goal: rostopic pub -1 /move_base_simple/goal "
                      "geometry_msgs/PoseStamped '{header: {frame_id: \"map\"}, "
                      "pose: {position: {x: 3.0, y: 0.0}, orientation: {w: 1.0}}}'")

    def cmd_cb(self, msg):
        self.twist = msg
        self.last_cmd_time = rospy.Time.now()

    def raycast(self, angle):
        step = self.resolution * 0.5
        x, y = self.x, self.y
        ca, sa = math.cos(angle), math.sin(angle)
        dist = 0.0
        while dist < self.range_max:
            x += step * ca
            y += step * sa
            dist += step
            col = int((x - self.origin_x) / self.resolution)
            row = int((y - self.origin_y) / self.resolution)
            if col < 0 or col >= self.width or row < 0 or row >= self.height:
                return float("inf")
            if self.occupied[self.height - 1 - row, col]:
                return dist
        return float("inf")

    def scan_timer(self, _event):
        msg = LaserScan()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "laser"
        msg.angle_min = -math.pi
        msg.angle_increment = 2.0 * math.pi / self.beam_count
        msg.angle_max = msg.angle_min + (self.beam_count - 1) * msg.angle_increment
        msg.range_min = 0.05
        msg.range_max = self.range_max
        msg.ranges = [self.raycast(self.yaw + msg.angle_min + i * msg.angle_increment)
                      for i in range(self.beam_count)]
        self.scan_pub.publish(msg)

    def odom_timer(self, _event):
        now = rospy.Time.now()
        dt = 1.0 / 30.0
        v = self.twist.linear.x
        w = self.twist.angular.z
        if (now - self.last_cmd_time).to_sec() > 0.5:
            v = 0.0
            w = 0.0
        mid = self.yaw + 0.5 * w * dt
        self.x += v * math.cos(mid) * dt
        self.y += v * math.sin(mid) * dt
        self.yaw += w * dt

        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)
        self.tf_broadcaster.sendTransform(
            (self.x, self.y, 0.0), q, now, "base_footprint", "odom_combined")

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom_combined"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.pose.covariance[0] = 0.01
        odom.pose.covariance[7] = 0.01
        odom.pose.covariance[35] = 0.01
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("fake_robot")
    FakeRobot()
    rospy.spin()
