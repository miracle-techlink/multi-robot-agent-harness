# multi-robot-agent-harness
# T1 Pro 端侧视觉建图全栈系统 (cuVSLAM + TOF Occupancy)

本仓库实现了在 NVIDIA Jetson Orin Nano (8GB) 上运行的本地视觉建图系统。系统仅依赖纯双目视觉和 TOF 点云，无需云端，即可在端侧实时完成位姿估计、2D Occupancy 栅格建图、拓扑图构建以及 ORB 视觉地标提取。

## ⚙️ 核心特性
- **纯视觉位姿估计**：基于 cuVSLAM `Multicamera` 模式，输出 10Hz 实时位姿（带去畸变处理）。
- **TOF 直连建图**：摒弃易受纹理影响的视觉视差深度，直接解析 10Hz、单帧 54720 点的 TOF 点云，生成高精度的 2D 栅格。
- **射线清空算法**：基于 Bresenham 算法进行 free space（空闲区）更新，完整输出 occupied/free/unknown 三态。
- **拓扑与地标**：轨迹每移动 0.5m 触发一次拓扑节点采样，并提取 ORB 特征点保存为视觉地标。
- **世界偏移防漂移（核心专利级逻辑）**：当机器人剧烈运动（如起立/趴下）导致纯视觉崩溃时，系统自动重建 Tracker。通过“世界偏移法”，将坐标锚定在最后一次正确的物理位置，**保证地图绝不碎裂、拓扑绝对连续**。

## ⚠️ 硬件层的已知限制 (Known Limitations)
1. **底层相机错帧**：实测左右目存在 100ms 级别的丢帧错位，代码内已加入严格的时间戳守卫（>5ms 直接丢弃），因此有效帧率约为 5-7Hz。
2. **IMU 缺失**：经排查，底盘硬件未提供有效的相机-IMU 外参标定（标定文件中全为 NaN）。因此本系统采用**纯视觉运行 + 底盘 Odom 平滑补偿**的策略，不使用 IMU 融合。

## 🚀 运行依赖
- cuVSLAM == 17.0.1 (ARM64 wheel)
- OpenCV (带 Python bindings)
- ROS 2 Humble

## 📂 输出产物
按 `Ctrl+C` 退出后，将自动在 `/robot/persist/t1_project/` 下保存：
- `rt_occupancy.png`: 包含 free/occupied/unknown 的 2D 栅格地图
- `rt_topology.json`: 轨迹拓扑节点
- `rt_landmarks.npz`: ORB 视觉地标描述子
