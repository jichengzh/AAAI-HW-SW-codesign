#!/bin/bash
set +u
source /opt/ros/noetic/setup.bash
source ~/ROS1/wheeltec_robot/devel/setup.bash
cleanup() {
  timeout 3 rostopic pub -1 /cmd_vel geometry_msgs/Twist "{}" >/dev/null 2>&1
  rosnode kill -a >/dev/null 2>&1
  kill -INT $LAUNCH_PID 2>/dev/null
  sleep 2
  pkill -x rosmaster 2>/dev/null
}
trap cleanup EXIT
roslaunch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch car_mode:=senior_akm > /tmp/uniad_base.log 2>&1 &
LAUNCH_PID=$!
sleep 6
roslaunch uniad_bridge uniad_bridge.launch > /tmp/uniad_bridge.log 2>&1 &
sleep 4
V2=/home/jichengzhi/liruozhan/toOrinV2
INPUT=${INPUT:-$V2/uniad_trt_input_119plus}
FRAMES=${FRAMES:-121}
echo "=== 启动 uniad_udp ${FRAMES}帧推理（桥接禁用，无动轮指令）==="
APP=$V2/DL4AGX_min/uniad-trt/inference_app/enqueueV2
cd $APP
UNIAD_SKIP_VIS=1 timeout 150 ./build/uniad_udp \
  $V2/uniad_tiny_fp16.engine ./build/libuniad_plugin.so \
  $INPUT $V2/output_udp $FRAMES > /tmp/uniad_app.log 2>&1 &
APP_PID=$!
sleep 45
echo "=== BRIDGE STATUS ==="
timeout 3 rostopic echo -n1 /uniad_bridge/status 2>&1
echo "=== PLANNED PATH (前2路点) ==="
timeout 3 rostopic echo -n1 /uniad_bridge/planned_path 2>&1 | head -25
echo "=== CMD_VEL (应为0) ==="
timeout 3 rostopic echo -n1 /cmd_vel 2>&1 | grep -A3 linear
wait $APP_PID
echo "APP EXIT: $?"
echo "=== APP LOG TAIL ==="
tail -4 /tmp/uniad_app.log
echo "=== BRIDGE NaN/FAULT ==="
grep -iE 'nan|fault|error' /tmp/uniad_bridge.log | head -8
echo "=== DRYRUN DONE ==="
