cat << 'EOF' > /robot/persist/t1_project/stop_vslam.sh
#!/bin/bash
echo "=== 停止 T1 VSLAM Node ==="
pkill -f rt_node.py
echo "已停止。建议执行: sudo reboot 恢复机器人原有业务。"
EOF
chmod +x /robot/persist/t1_project/stop_vslam.sh
