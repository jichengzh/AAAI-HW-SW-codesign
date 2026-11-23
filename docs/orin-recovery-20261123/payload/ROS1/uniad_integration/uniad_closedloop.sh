#!/bin/bash
set +u
source /opt/ros/noetic/setup.bash
source ~/ROS1/wheeltec_robot/devel/setup.bash
cleanup() {
  timeout 3 rostopic pub -1 /cmd_vel geometry_msgs/Twist "{}" >/dev/null 2>&1
  sleep 1
  rosnode kill -a >/dev/null 2>&1
  kill -INT $LAUNCH_PID 2>/dev/null
  sleep 2
  pkill -x rosmaster 2>/dev/null
}
trap cleanup EXIT
roslaunch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch car_mode:=senior_akm > /tmp/uniad_cl_base.log 2>&1 &
LAUNCH_PID=$!
sleep 6
roslaunch uniad_bridge uniad_bridge.launch enable:=true max_speed:=0.15 > /tmp/uniad_cl_bridge.log 2>&1 &
sleep 4
echo "=== cmd_vel/odom 采样器启动（45秒）==="
timeout 45 /usr/bin/python3 -u - <<'PY' > /tmp/uniad_cl_samples.txt 2>&1 &
import rospy, time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
rospy.init_node('cl_sampler', anonymous=True)
st = {'maxv':0.0,'maxw':0.0,'n':0,'nz':0,'x0':None,'x':0.0,'y':0.0}
def cb(m):
    v,w = m.linear.x, m.angular.z
    st['maxv']=max(st['maxv'],abs(v)); st['maxw']=max(st['maxw'],abs(w))
    st['n']+=1; st['nz']+= (abs(v)>1e-4 or abs(w)>1e-4)
def odom(m):
    if st['x0'] is None: st['x0']=m.pose.pose.position.x
    st['x']=m.pose.pose.position.x; st['y']=m.pose.pose.position.y
rospy.Subscriber('/cmd_vel',Twist,cb); rospy.Subscriber('/odom',Odometry,odom)
t0=time.time()
while time.time()-t0<43 and not rospy.is_shutdown(): time.sleep(0.5)
print(f"cmd_vel msgs={st['n']} nonzero={st['nz']} max|v|={st['maxv']:.3f} max|w|={st['maxw']:.3f}")
print(f"odom delta: x={st['x']-(st['x0'] or 0):+.3f} y={st['y']:+.3f}")
PY
SAMPLER_PID=$!
sleep 2
V2=/home/jichengzhi/liruozhan/toOrinV2
INPUT=${INPUT:-$V2/uniad_trt_input_119plus}
FRAMES=${FRAMES:-121}
echo "=== uniad_udp ${FRAMES}帧闭环推理开始 ==="
cd $V2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
UNIAD_SKIP_VIS=1 timeout 90 ./build/uniad_udp \
  $V2/uniad_tiny_fp16.engine ./build/libuniad_plugin.so \
  $INPUT $V2/output_udp $FRAMES > /tmp/uniad_cl_app.log 2>&1
echo "APP EXIT: $?"
wait $SAMPLER_PID 2>/dev/null
echo "=== 采样结果 ==="
cat /tmp/uniad_cl_samples.txt
echo "=== NaN 跳过记录 ==="
grep -i 'nan' /tmp/uniad_cl_bridge.log | head -5
echo "=== 最终状态 ==="
timeout 3 rostopic echo -n1 /uniad_bridge/status 2>&1
echo "=== CLOSEDLOOP DONE ==="
