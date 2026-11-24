# ZeroTier、SSH与启动环境的私有恢复

2026-11-24已将原Orin的25份配置文件流式备份到独立服务器bm-2s55he，逐文件SHA256核对并在独立临时目录实际还原、重新核对通过。传输使用现有SSH；未改动原网络或SSH服务。公开状态见 [备份状态](../validation/private_configuration_backup_20261124.json)。

**备份实体只在本地服务器，不在GitHub。** 其中包含ZeroTier身份、SSH用户私钥和服务器host keys，目录权限0700、文件0600；不要将这些实体或其内容放入Git、CI日志或聊天。

```text
/home/jichengzhi/.local/share/orin-recovery-private/20261124/
├── orin-user-network-config.tar.gz
├── orin-system-network-config.tar.gz
├── *.manifest.json
├── SHA256SUMS
├── backup_receipt.json
└── restore_check_receipt.json
```

用户包包含原用户.ssh、.bashrc与.profile的普通文件；系统包包含ZeroTier身份/网络成员配置、SSH客户端/sshd配置、host keys及实际存在的相关配置文件。临时socket、软件包提供的服务启用链接和无关的用户服务软链接不作为配置文件还原；安装软件和systemctl enable会重建服务链接。每份包内有私有manifest，保存相对路径、大小、摘要和原权限。shell历史、完整环境变量和实验缓存不在此包。

## 先建立恢复入口

通过显示器/串口、本地有线网络或其它可用入口完成新系统安装，再将两份私有包及其SHA256SUMS传到新Orin的受限目录。**不能依赖尚未恢复的ZeroTier地址来传入它自己的配置。**

原软件版本为zerotier-one 1.16.2、openssh-server 1:8.2p1-4ubuntu0.13。按[ZeroTier官方安装说明](https://www.zerotier.com/download/)配置适合focal/arm64的安装源，再查询并安装；精确旧版本不可取时记录实际版本并重新验证，不把版本账本当作已安装结果：

```bash
sudo apt-get update
apt-cache policy zerotier-one openssh-server
sudo apt-get install zerotier-one openssh-server
```

ZeroTier身份和成员配置位于[官方说明的服务目录](https://docs.zerotier.com/config/)；恢复旧身份后应只有恢复目标使用该身份在线运行。以下在**原机已离线的新恢复目标**执行，不能在仍联网的原机或另一台常用服务器上试启动相同身份。

## 校验并暂存文件，不直接覆盖新系统

```bash
umask 077
export PRIVATE=/home/jichengzhi/orin-recovery/private_configuration
chmod 700 "$PRIVATE"
cd "$PRIVATE"
sha256sum -c SHA256SUMS
export PRIVATE_STAGE
PRIVATE_STAGE=$(mktemp -d "$PRIVATE/restore-stage.XXXXXX")
tar -xzf "$PRIVATE/orin-user-network-config.tar.gz" -C "$PRIVATE_STAGE"
tar -xzf "$PRIVATE/orin-system-network-config.tar.gz" -C "$PRIVATE_STAGE"
```

暂存目录中的`home/jichengzhi/.ssh`、`var/lib/zerotier-one`和`etc/ssh`为原路径布局。包内manifest包含敏感文件的摘要，仅保留在私有目录；不需要输出私钥或token以验证备份。

## 恢复ZeroTier与SSH

新系统上先将安装产生的网络/SSH配置另存到私有目录，以便回滚。确认暂存文件已校验后：

```bash
# 先检查新安装包已生成的状态文件，沿用其账号名而非旧数值UID。
ZT_STATE_USER=$(sudo stat -c '%U' /var/lib/zerotier-one/identity.secret)
ZT_STATE_GROUP=$(sudo stat -c '%G' /var/lib/zerotier-one/identity.secret)
test -n "$ZT_STATE_USER" && test -n "$ZT_STATE_GROUP"
sudo systemctl stop zerotier-one
sudo install -d -m 700 /var/lib/zerotier-one
sudo cp -a "$PRIVATE_STAGE/var/lib/zerotier-one/." /var/lib/zerotier-one/
sudo chown -R "$ZT_STATE_USER:$ZT_STATE_GROUP" /var/lib/zerotier-one
sudo chmod 700 /var/lib/zerotier-one
sudo find /var/lib/zerotier-one -type f -name '*.secret' -exec chmod 600 {} +

install -d -m 700 /home/jichengzhi/.ssh
cp -a "$PRIVATE_STAGE/home/jichengzhi/.ssh/." /home/jichengzhi/.ssh/
sudo chown -R jichengzhi:jichengzhi /home/jichengzhi/.ssh
chmod 700 /home/jichengzhi/.ssh
find /home/jichengzhi/.ssh -type f -exec chmod 600 {} +
```

原1.16.2实际将identity文件归 `zerotier-one` 账号，旧UID/GID为999/993；systemd的User字段虽为空，进程仍会自行降权，因此不能仅凭该字段把状态文件改成root所有。上面从新安装包生成的identity文件读取实际账号名，再恢复和重设所有权。若该文件尚未生成，先完成软件安装与初始化检查后再恢复；不要沿用已失效的数值UID。

为了保留原SSH服务器身份，可在备份新系统host keys后，将暂存`etc/ssh`中的原host keys、sshd配置及实际存在的drop-in放回对应路径。用`cp -a`保留原文件权限，私有host keys必须root所有、0600；不要仅为了沿用指纹而覆盖新系统未核对的全部SSH配置。完成修改后先校验再重载：

```bash
sudo /usr/sbin/sshd -t
sudo systemctl enable --now ssh
sudo systemctl reload ssh
sudo systemctl enable --now zerotier-one
sudo zerotier-cli info
sudo zerotier-cli listnetworks
```

`listnetworks`显示网络ONLINE/OK后，从本地服务器另开一个SSH连接确认用户认证与端点可达，再关闭原维护连接。控制器成员授权、分配地址和路由仍须与原网络核对；若控制器配置已改变，恢复节点身份不会撤销外部变更。参考[ZeroTier节点身份说明](https://docs.zerotier.com/protocol/)。

## 恢复shell设置

先比对暂存`.bashrc`/`.profile`，在ROS、conda、CUDA和项目目录已经重建后合并对应source及动态库路径设置。不要在安装依赖前直接source旧文件；旧路径需与[安装目录](../DIRECTORY_MAP.md)核对。保存整个文件用于比较，并不要求覆盖新系统默认文件。

此步骤恢复远程入口和启动设置，不替代Copt/UniAD服务验收。完整Copt参数衔接见 [服务恢复](../COPT_SERVICE_RESTORE.md)，联网按版本安装依赖见 [环境说明](ENVIRONMENT_RESTORE.md)。
