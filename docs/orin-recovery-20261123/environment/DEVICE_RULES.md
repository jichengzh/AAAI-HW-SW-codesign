# 小车串口、雷达和相机规则恢复

2026-11-23 从 Orin 的 `/etc/udev/rules.d/` **只读复制15个相关 `.rules` 文本文件，共14,435字节**，内容未修改；文件SHA256、原路径、源码来源和用户设备相关组名见 [device_rules/manifest.json](device_rules/manifest.json)。未采集passwd/shadow、网络密码、SSH配置或其他无关规则。未在远端执行安装、udev重载、设备触发或用户组修改。

## 已保存内容和匹配条件

| 文件 | 恢复的设备别名/用途 |
|---|---|
| `wheeltec_controller.rules`、`wheeltec_controller2.rules`、`wheeltec_controller3.rules` | `/dev/wheeltec_controller`；USB串口serial=`0002` |
| `wheeltec_lidar.rules`、`wheeltec_lidar2.rules`、`wheeltec_lidar3.rules` | `/dev/wheeltec_lidar`；serial=`0001` |
| `wheeltec_imu.rules`、`wheeltec_imu_343.rules`、`wheeltec_imu_ACM.rules` | `/dev/wheeltec_IMU`；serial=`0003` |
| `wheeltec_gnss.rules` | `/dev/wheeltec_gnss`；CH9102/ACM serial=`0005` |
| `wheeltec_mmwave_radar.rules`、`wheeltec_mmwave_radar2.rules`、`wheeltec_mmwave_radar3.rules` | `/dev/wheeltec_mmwave_radar`；serial=`0006` |
| `camera.rules` | 按相机名称/VID/PID建立`RgbCam`、`Astra_Pro`、`Astra_Dabai`、`Astra_Gemini`、`realsense`等别名 |
| `56-orbbec-usb.rules` | Orbbec USB VID=`2bc5`的Astra等型号访问规则 |

串口规则同时覆盖CP2102（VID:PID=`10c4:ea60`，`ttyUSB*`）和部分CH9102（`1a86:55d4`，`ttyCH343USB*`或`ttyACM*`）路径；具体范围以各规则原文为准。这里的`0001/0002/...`是匹配设备提供的serial属性，不是凭据。更换设备、USB转换器、设备serial或驱动后，应检查实际属性，不能仅改别名就认为设备匹配成功。udev规则本身不会安装CH343等内核驱动。

**保留的原厂权限策略：**这些历史规则把部分设备设置为`0666`或`0777`，即允许本机其他用户访问匹配设备；本快照原样保留，不把它描述成仅dialout组可访问。若后续改为`0660`及指定组，应另存修订规则并重新做设备访问验收，保留当前原始快照供核对。

`99-tegra-devices.rules`属于Jetson BSP的广泛设备配置，本次排除；由匹配的Jetson系统映像/系统包恢复，不从本目录覆盖。

## 在格式化后的Orin安装这15个规则

以下命令**会写入新系统**，本次采集没有执行。先停止底盘/传感器相关业务，确认恢复仓库路径；无需运行厂商脚本或启动ROS控制节点。

```bash
export RECOVERY_ENV=/home/jichengzhi/orin-recovery/github/environment
export RULES_DIR="$RECOVERY_ENV/device_rules"

# 先只读校验本地文件哈希。
python3 -B - <<'PY'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ['RULES_DIR'])
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest['rules']:
    path = root / entry['filename']
    raw = path.read_bytes()
    assert len(raw) == entry['bytes'], path
    assert hashlib.sha256(raw).hexdigest() == entry['sha256'], path
print('15 device rule files verified')
PY

# 只安装明确列出的15个文件；不复制整个/etc或运行原厂安装脚本。
rule_files=(
  56-orbbec-usb.rules camera.rules
  wheeltec_controller.rules wheeltec_controller2.rules wheeltec_controller3.rules
  wheeltec_gnss.rules wheeltec_imu.rules wheeltec_imu_343.rules wheeltec_imu_ACM.rules
  wheeltec_lidar.rules wheeltec_lidar2.rules wheeltec_lidar3.rules
  wheeltec_mmwave_radar.rules wheeltec_mmwave_radar2.rules wheeltec_mmwave_radar3.rules
)
sudo install -d -m 0755 /etc/udev/rules.d
for rule_file in "${rule_files[@]}"; do
  sudo install -m 0644 -o root -g root "$RULES_DIR/$rule_file" "/etc/udev/rules.d/$rule_file"
done
sudo udevadm control --reload-rules
```

`install -m0644`设置的是**规则文本文件**的权限，不会把规则内部的设备权限改为0644。规则重载后，在相关业务停止的情况下重新插拔目标USB设备，使其重新枚举并应用规则；本步骤不调用全局`udevadm trigger`，不启动底盘/闭环/推理。

## 恢复用户访问组

原用户`jichengzhi`的设备相关组为 **dialout、video、plugdev、i2c、gpio**（仅保存这些allowlist组名，没有保存用户/组数据库）。串口通常用dialout；视频/USB设备相关访问用video、plugdev。新系统可按当前要连接的设备恢复：

```bash
# 新系统应已存在这些标准组，先只打印相应组名。
getent group dialout video plugdev | cut -d: -f1
sudo usermod -aG dialout,video,plugdev jichengzhi
```

必须保留`-aG`中的`-a`，避免替换用户原有附加组。执行后**完全注销当前图形/终端/SSH会话并重新登录**，或重启，再确认新会话组成员身份；已有ROS/容器/终端进程不会自动获得新组。`i2c`、`gpio`属于原Jetson设备访问状态，只有相应BSP组存在且业务需要时再加入，并非串口/相机恢复的前置条件：

```bash
# 可选：先确认组存在；仅在业务确实需要访问I2C/GPIO时执行。
getent group i2c gpio | cut -d: -f1
sudo usermod -aG i2c,gpio jichengzhi

# 重新登录后的只读检查：
id -nG | tr ' ' '\n' | sed -n '/^dialout$/p; /^video$/p; /^plugdev$/p; /^i2c$/p; /^gpio$/p'
ls -l /dev/wheeltec_controller /dev/wheeltec_lidar /dev/wheeltec_IMU 2>/dev/null
ls -l /dev/Astra_Pro /dev/Astra_Dabai /dev/Astra_Gemini /dev/RgbCam 2>/dev/null
```

设备未插入或匹配条件不同，别名可能不存在；这时先读实际节点属性，例如`udevadm info --attribute-walk --name=/dev/ttyUSB0`（替换为实际节点）。别名存在只证明节点和规则匹配，**不证明串口数据、激光雷达帧、图像或底盘控制已验收**。此处不提供发送串口命令或启动运动的测试命令。

## 厂商源码中的备用来源

- `56-orbbec-usb.rules`与下面两份原文件SHA256完全一致：`~/ROS1/wheeltec_robot/src/ros_astra_camera-main/56-orbbec-usb.rules`、`~/ROS1/vendor/src/ros_astra_camera-main/56-orbbec-usb.rules`。因此该相机规则也可由匹配的厂商源码恢复。
- `~/ROS1/wheeltec_robot/src/turn_on_wheeltec_robot/wheeltec_udev.sh`及`~/ROS1/vendor/src/turn_on_wheeltec_robot/wheeltec_udev.sh`中包含已安装controller、lidar、IMU、GNSS、毫米波雷达和camera规则的生成行。`scripts/wheeltec_udev.sh`、`ldlidar_ros1/ldlidar_udev.sh`等还包含部分匹配行。路径、脚本SHA256和逐项匹配见manifest。
- 这些生成器有不同版本，可能额外创建规则、追加重复行、重载/重启udev；源码中存在相同生成行，不等于执行任一脚本都会只恢复当前15个文件。**本次恢复优先安装本目录已核对的规则文本**。源码/厂商包是否已收录到payload仍以`manifests/source_capture_manifest.json`为准；原远端vendor目录不等于恢复仓库完整保存了vendor树。
