cat << 'EOF' > /robot/persist/t1_project/start_vslam.sh
#!/bin/bash
echo "=================================================="
echo "⚠️ 请确保机器人处于急停/锁定状态！"
echo "⚠️ 当前操作将停止 percep、aimrt_vslam 等节点释放 GPU"
echo "=================================================="
read -p "确认继续吗？(y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "已取消。"
    exit 1
fi

echo "=== 1. 停止重负载业务节点 ==="
pkill -f perception/bin/percep
pkill -f target_follow_perception
pkill -f aimrt_vslam
sleep 2

echo "=== 2. 启动 T1 VSLAM ROS 2 节点 ==="
source /opt/ros/humble/setup.bash
python3 /robot/persist/t1_project/rt_node.py

echo "=== 3. VSLAM 节点已退出 ==="
echo "为了保证原有业务完全恢复，建议执行: sudo reboot"
EOF
chmod +x /robot/persist/t1_project/start_vslam.sh
