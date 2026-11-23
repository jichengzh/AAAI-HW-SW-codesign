#!/usr/bin/python3
"""UniAD planning (UDP from uniad_udp app) -> /cmd_vel with safety guards.

Packet: int32 frame_id, int32 command, float32[12] (6 waypoints x,y in ego frame).
Safety: explicit enable service, NaN guard (latched), watchdog, speed/steer caps.
"""
import math
import socket
import struct
import threading

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse

PACKET = struct.Struct('<ii12f')


class Bridge:
    def __init__(self):
        self.max_speed = rospy.get_param('~max_speed', 0.15)
        self.wheelbase = rospy.get_param('~wheelbase', 0.322)
        self.max_steer = rospy.get_param('~max_steer', math.atan(self.wheelbase / 0.8))
        self.steer_gain = rospy.get_param('~steer_gain', 1.0)
        self.lookahead = rospy.get_param('~lookahead', 0.8)
        self.watchdog = rospy.get_param('~watchdog_timeout', 1.5)
        self.stop_dist = rospy.get_param('~stop_distance', 0.15)
        self.port = rospy.get_param('~udp_port', 9876)

        self.enabled = rospy.get_param('~enable', False)
        self.nan_count = 0
        self.traj = None
        self.frame_id = -1
        self.last_rx = 0.0
        self.lock = threading.Lock()

        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.pub_path = rospy.Publisher('~planned_path', Path, queue_size=5)
        self.pub_status = rospy.Publisher('~status', String, queue_size=10)
        rospy.Service('~set_enable', SetBool, self.on_enable)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', self.port))
        self.sock.settimeout(0.1)
        threading.Thread(target=self.rx_loop, daemon=True).start()
        rospy.Timer(rospy.Duration(0.05), self.control_loop)
        rospy.loginfo('uniad_bridge up: enable=%s max_speed=%.2f max_steer=%.3f',
                      self.enabled, self.max_speed, self.max_steer)

    def on_enable(self, req):
        with self.lock:
            self.enabled = req.data
            if req.data:
                self.nan_count = 0
        rospy.logwarn('bridge %s', 'ENABLED' if req.data else 'DISABLED')
        return SetBoolResponse(success=True, message='enabled' if req.data else 'disabled')

    def rx_loop(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            if len(data) != PACKET.size:
                continue
            fid, cmd, *pts = PACKET.unpack(data)
            # engine outs_planning axis order: (lateral right+, forward); internal: (x forward, y left+)
            traj = [(pts[i + 1], -pts[i]) for i in range(0, 12, 2)]
            with self.lock:
                self.frame_id = fid
                self.last_rx = rospy.Time.now().to_sec()
                if any(math.isnan(v) or math.isinf(v) for p in traj for v in p):
                    # NaN frame: discard, command zero until next clean frame
                    self.nan_count += 1
                    self.traj = None
                    rospy.logwarn('NaN/Inf planning frame %d skipped (total %d)', fid, self.nan_count)
                else:
                    self.traj = traj
            self.publish_path(traj)

    def publish_path(self, traj):
        p = Path()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = 'base_footprint'
        for x, y in traj:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        self.pub_path.publish(p)

    def control_loop(self, _event):
        now = rospy.Time.now().to_sec()
        cmd = Twist()
        with self.lock:
            traj, enabled = self.traj, self.enabled
            stale = (now - self.last_rx) > self.watchdog
        if not enabled:
            status = 'disabled (call ~set_enable true)'
        elif traj is None or stale:
            status = 'waiting/stale/NaN trajectory -> stop'
        else:
            v, delta, status = self.pure_pursuit(traj)
            cmd.linear.x = v
            cmd.angular.z = v * math.tan(delta) / self.wheelbase
        self.pub_cmd.publish(cmd)
        self.pub_status.publish(String(data=status))

    def pure_pursuit(self, traj):
        dists = [math.hypot(x, y) for x, y in traj]
        if max(dists) < self.stop_dist:
            return 0.0, 0.0, 'trajectory ~stationary -> stop'
        target = None
        if dists[0] > self.lookahead:
            # waypoints are sparse (~2s spacing); interpolate on segment ego->wp0
            x0, y0 = traj[0]
            k = self.lookahead / dists[0]
            target = (x0 * k, y0 * k, self.lookahead)
        else:
            for (x, y), d in zip(traj, dists):
                if d >= self.lookahead:
                    target = (x, y, d)
                    break
        if target is None:
            x, y = traj[-1]
            target = (x, y, dists[-1])
        x, y, ld = target
        ld = max(ld, 0.1)
        kappa = 2.0 * y / (ld * ld)
        delta = math.atan(self.wheelbase * kappa)
        delta = max(-self.max_steer, min(self.max_steer, delta))
        v = self.max_speed / (1.0 + 2.0 * abs(delta) / self.max_steer)
        v = max(0.0, min(self.max_speed, v))
        # steer_gain: visualization amplifier, applied after v so speed profile is unchanged
        delta *= self.steer_gain
        delta = max(-self.max_steer, min(self.max_steer, delta))
        return v, delta, 'frame %d: v=%.3f delta=%.3f' % (self.frame_id, v, delta)


if __name__ == '__main__':
    rospy.init_node('uniad_cmdvel_bridge')
    Bridge()
    rospy.spin()
