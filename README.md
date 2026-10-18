# multi-robot-agent-harness

# T1 Pro 端侧视觉建图全栈系统 (cuVSLAM + TOF Occupancy)

本仓库实现了在 NVIDIA Jetson Orin Nano (8GB) 上运行的本地视觉建图系统。系统仅依赖纯双目视觉和 TOF 点云，无需云端，即可在端侧实时完成位姿估计、2D Occupancy 栅格建图、拓扑图构建以及 ORB 视觉地标提取，并**以标准 ROS 2 消息对外发布，可直接接入 Nav2 导航系统**。

## ⚙️ 核心特性
- **纯视觉位姿估计**：基于 cuVSLAM `Multicamera` 模式，输出 10Hz 实时位姿（带去畸变处理）。
- **TOF 直连建图与射线清空**：摒弃易受纹理影响的视觉视差深度，直接解析 10Hz、单帧 54720 点的 TOF 点云。基于 Bresenham 算法进行 free space（空闲区）更新，完整输出 occupied/free/unknown 三态。
- **拓扑与地标生成**：轨迹每移动 0.5m 触发一次拓扑节点采样，并提取 ORB 特征点保存为视觉地标。
- **世界偏移防漂移锁（核心逻辑）**：当机器人剧烈运动（如起立/趴下）导致纯视觉崩溃时，系统自动重建 Tracker。通过“世界偏移法”结合底盘 `/odom` 增量补偿，将坐标锚定在最后一次正确的物理位置，**保证地图绝不碎裂、拓扑绝对连续、系统绝不死锁**。
- **标准 ROS 2 接口**：发布标准 `nav_msgs/Odometry`、`nav_msgs/OccupancyGrid` 及 `TF` 树，无缝对接上层业务。

## 📡 业务接口 (ROS 2 Topics 和 TF)
| 话题名 | 类型 | 频率 | 说明 |
| :--- | :--- | :--- | :--- |
| `/rt_odometry` | `nav_msgs/Odometry` | 10 Hz | 实时位姿 |
| `/rt_occupancy_grid` | `nav_msgs/OccupancyGrid` | 5 Hz | 2D 栅格地图（-1:未知, 0:空闲, 100:障碍物）|
| `TF` | `tf2_msgs/TFMessage` | 10 Hz | `map` -> `camera_link` |

## ⚠️ 硬件层的已知限制 (Known Limitations)
1. **底层相机错帧**：实测左右目存在 100ms 级别的偶发丢帧错位。代码内已加入严格的时间戳守卫（>5ms 直接丢弃），因此有效建图帧率约为 5-7Hz。
2. **IMU 缺失**：经排查，底盘硬件未提供有效的相机-IMU 外参标定（标定文件中全为 NaN）。因此本系统采用**纯视觉运行 + 底盘 Odom 平滑补偿**的退阶策略，不依赖 IMU。
3. **长距离尺度漂移**：受限于纯视觉累积误差，剧烈运动下仍存在尺度漂移，但已通过防漂移锁保证系统不崩溃、地图不撕裂。

## 🚀 运行依赖
- **硬件**: NVIDIA Jetson Orin Nano (JetPack 6.x, L4T R36.4.4)
- **依赖库**: 
  - ROS 2 Humble
  - OpenCV (带 Python bindings)
  - cuVSLAM == 17.0.1 (需安装 ARM64 预编译 wheel)

## 🛠️ 快速启动 (一键启停脚本)
由于本模块占用较高的 GPU 资源，启动前需切断原有的重负载感知节点（`percep`、`aimrt_vslam` 等）。我们提供了一键脚本：

**1. 启动建图模块**
```bash
# 运行前请确保机器人处于急停/锁定状态
/robot/persist/t1_project/start_vslam.sh
2. 停止建图模块

bash
/robot/persist/t1_project/stop_vslam.sh
# 退出后，建议执行 sudo reboot 彻底恢复原有业务节点
(注：代码及脚本路径硬编码为 /robot/persist/t1_project/，若在其他机器上运行请注意修改。)

📊 可视化验证 (Foxglove)
推荐使用 Foxglove Studio 进行远程可视化（支持 Windows 端）：

在 Nano 上启动 WebSocket 桥接服务：ros2 launch rosbridge_server rosbridge_websocket_launch.xml

在 Windows 上建立 SSH 隧道：ssh -L 9090:127.0.0.1:9090 run@<Nano_IP>

打开 Foxglove 连接 ws://127.0.0.1:9090，在 3D 面板添加 Map 图层，绑定 /rt_occupancy_grid，将 Fixed Frame 设为 map 即可查看实时建图。

## 📂 离线输出产物 (运行 rt_final.py)
若通过修改代码或运行早期版本脚本，按 Ctrl+C 退出后，会在本地保存以下产物文件：

rt_occupancy.png: 包含 free/occupied/unknown 的 2D 栅格地图

rt_topology.json: 轨迹拓扑节点记录

rt_landmarks.npz: ORB 视觉地标描述子

## 🔮 后续优化方向 (Roadmap)
修复底层硬同步：建议联系硬件/驱动团队修复双目 100ms 错帧，恢复硬件触发，可成倍提升建图精度。

## Odom 紧耦合：研究 cuVSLAM 的 External Odometry 接口，将底盘 Odom 作为底层输入，彻底解决长距离漂移问题。

ROS 2 规范化封装：将单文件 Python 脚本拆分为标准的 ROS 2 Package（包含 setup.py, config, launch），并实现 systemd 开机自启。
