import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cuvslam
import time
import cv2
import json

def get_world_pose(pose_estimate):
    if pose_estimate is None or pose_estimate.world_from_rig is None:
        return None
    pose = pose_estimate.world_from_rig.pose
    q, t = pose.rotation, pose.translation
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

class RealtimeFinal(Node):
    def __init__(self):
        super().__init__('realtime_final')
        
        # 1. 相机配置
        scale = 0.2469
        self.fx, self.fy = 1070.14 * scale, 1070.21 * scale
        self.cx, self.cy = 1298.54 * scale, 964.829 * scale
        self.baseline = 0.0598

        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        D = np.array([0.009450, -0.006347, 0.0, 0.0, -0.007494, 0.002330, 0.0, 0.0], dtype=np.float32)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, K, (640, 480), cv2.CV_32FC1)

        self.cam0 = cuvslam.Camera(); self.cam0.size = (640, 480); self.cam0.focal = (self.fx, self.fy); self.cam0.principal = (self.cx, self.cy)
        self.cam0.rig_from_camera = cuvslam.Pose([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.cam1 = cuvslam.Camera(); self.cam1.size = (640, 480); self.cam1.focal = (self.fx, self.fy); self.cam1.principal = (self.cx, self.cy)
        self.cam1.rig_from_camera = cuvslam.Pose([1.0, 0.0, 0.0, 0.0], [-self.baseline, 0.0, 0.0])
        
        self.rig = cuvslam.Rig([self.cam0, self.cam1])
        self.cfg = cuvslam.Tracker.OdometryConfig()
        self.cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Multicamera
        self.tracker = cuvslam.Tracker(self.rig, self.cfg)
        
        # 2. 建图配置
        self.grid_size, self.res = 400, 0.05
        self.grid = np.full((self.grid_size, self.grid_size), 127, dtype=np.int8)
        self.origin_offset = self.grid_size // 2
        
        self.orb = cv2.ORB_create(nfeatures=300)
        
        # 3. 状态缓存
        self.last_l, self.last_r = None, None
        self.latest_tof_pts = None
        self.frame_count = 0
        self.start_time = time.time()
        
        self.last_valid_pos = None 
        self.world_offset = np.zeros(3) # 用于重置时保持坐标连续性
        self.hard_limit = 0.35  # 超过 0.35m 视为跳变
        self.consecutive_soft_limit = 0

        self.last_topo_pos = None
        self.topology_nodes = []
        self.landmarks = []
        self.dropped_frames = 0

        # Odom 缓存
        self.latest_odom_2d = None

        # 4. 订阅
        self.sub_l = self.create_subscription(Image, '/aima/hal/camera/head_stereo_left_mono', self.left_cb, 10)
        self.sub_r = self.create_subscription(Image, '/aima/hal/camera/head_stereo_right_mono', self.right_cb, 10)
        self.sub_tof = self.create_subscription(PointCloud2, '/percep/tof/front/point_cloud', self.tof_cb, 10)
        
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, qos_best_effort)

        self.get_logger().info("视觉 + Odom + TOF 终极融合版已启动！")

    def odom_cb(self, msg):
        # 简单提取 X, Y 平面坐标（具体视 T1 坐标系而定，这里用作避障参考）
        p = msg.pose.pose.position
        self.latest_odom_2d = np.array([p.x, p.y])

    def tof_cb(self, msg):
        self.latest_tof_pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)

    def left_cb(self, msg):
        self.last_l = msg
        self.process()

    def right_cb(self, msg):
        self.last_r = msg
        self.process()

    def reset_tracker(self):
        # 重置追踪器，但保留世界偏移
        if self.last_valid_pos is not None:
            self.world_offset = self.last_valid_pos.copy()
            self.get_logger().error(f"🚨 视觉跳变，重建 Tracker，世界坐标保持于 {self.world_offset}")
        else:
            self.world_offset = np.zeros(3)
        self.tracker = cuvslam.Tracker(self.rig, self.cfg)
        self.consecutive_soft_limit = 0

    def process(self):
        if self.last_l is None or self.last_r is None:
            return
        
        msg_l, msg_r = self.last_l, self.last_r
        t_l = msg_l.header.stamp.sec + msg_l.header.stamp.nanosec * 1e-9
        t_r = msg_r.header.stamp.sec + msg_r.header.stamp.nanosec * 1e-9
        if abs(t_l - t_r) > 0.005:
            self.dropped_frames += 1
            self.last_l, self.last_r = None, None
            return

        self.last_l, self.last_r = None, None
        self.frame_count += 1

        l_raw = np.frombuffer(msg_l.data, dtype=np.uint8).reshape((480, 640))
        r_raw = np.frombuffer(msg_r.data, dtype=np.uint8).reshape((480, 640))
        l_np = cv2.remap(l_raw, self.map1, self.map2, cv2.INTER_LINEAR)
        r_np = cv2.remap(r_raw, self.map1, self.map2, cv2.INTER_LINEAR)

        timestamp_ns = int(msg_l.header.stamp.sec * 1e9 + msg_l.header.stamp.nanosec)

        try:
            pose_est, _ = self.tracker.track(timestamp_ns, [l_np, r_np])
        except Exception as e:
            self.get_logger().error(f"Tracker error: {e}")
            self.reset_tracker()
            return

        T_wc = get_world_pose(pose_est)
        if T_wc is None:
            return

        # 获取相对坐标，并加上世界偏移
        current_pos = T_wc[:3, 3] + self.world_offset

        # 防漂移逻辑
        if self.last_valid_pos is not None:
            jump_dist = np.linalg.norm(current_pos - self.last_valid_pos)
            
            if jump_dist > self.hard_limit:
                self.reset_tracker()
                return 
                
            elif jump_dist > 0.1: # 轻微跳变（限幅）
                self.consecutive_soft_limit += 1
                if self.consecutive_soft_limit >= 5:
                    self.reset_tracker()
                    return
                
                diff = current_pos - self.last_valid_pos
                direction = diff / jump_dist
                corrected_pos = self.last_valid_pos + direction * 0.1
                T_wc[:3, 3] = corrected_pos - self.world_offset # 回写内部
                current_pos = corrected_pos
            else:
                self.consecutive_soft_limit = 0

        self.last_valid_pos = current_pos.copy()
        T_wc[:3, 3] = current_pos

        # 拓扑与 ORB
        if self.last_topo_pos is None or np.linalg.norm(current_pos - self.last_topo_pos) > 0.5:
            node = {"id": len(self.topology_nodes), "pose": T_wc.tolist()}
            self.topology_nodes.append(node)
            self.last_topo_pos = current_pos.copy()
            
            kp, des = self.orb.detectAndCompute(l_np, None)
            if des is not None:
                for i, k in enumerate(kp):
                    self.landmarks.append({"node_id": node["id"], "u": float(k.pt[0]), "v": float(k.pt[1]), "descriptor": des[i].tolist()})

        # 2D 栅格建图 (使用 TOF 数据)
        if self.latest_tof_pts is not None and self.frame_count % 2 == 0:
            pts_cam = self.latest_tof_pts
            # 这里假设 TOF 点云在相机系下，直接使用 T_wc 投影
            valid_mask = (pts_cam[:, 2] > 0.2) & (pts_cam[:, 2] < 5.0)
            valid_pts = pts_cam[valid_mask]
            
            if len(valid_pts) > 0:
                pts_world = (T_wc[:3, :3] @ valid_pts.T).T + T_wc[:3, 3]
                wx, wy, wz = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]
                height_mask = (wy > -1.0) & (wy < 0.1)
                
                gx = (wx[height_mask] / self.res + self.origin_offset).astype(int)
                gy = (wz[height_mask] / self.res + self.origin_offset).astype(int)
                
                valid_g = (gx >= 0) & (gx < self.grid_size) & (gy >= 0) & (gy < self.grid_size)
                gx, gy = gx[valid_g], gy[valid_g]
                self.grid[gy, gx] = 100
                
                cam_gx = int(T_wc[0, 3] / self.res + self.origin_offset)
                cam_gy = int(T_wc[2, 3] / self.res + self.origin_offset)
                for px, py in zip(gx, gy):
                    cv2.line(self.grid, (cam_gx, cam_gy), (px, py), 0, 1)

        # 打印状态
        if self.frame_count % 10 == 0:
            fps = self.frame_count / (time.time() - self.start_time)
            t = current_pos
            self.get_logger().info(f"F{self.frame_count} | FPS:{fps:.1f} | Pos:[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}] | 拓扑:{len(self.topology_nodes)} | 丢帧:{self.dropped_frames}")

def main(args=None):
    rclpy.init(args=args)
    node = RealtimeFinal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("保存数据...")
        with open("/robot/persist/t1_project/rt_topology.json", "w") as f:
            json.dump({"nodes": node.topology_nodes}, f)
        np.savez("/robot/persist/t1_project/rt_landmarks.npz", landmarks=np.array(node.landmarks, dtype=object), allow_pickle=True)
        
        display_grid = np.zeros((node.grid_size, node.grid_size), dtype=np.uint8)
        display_grid[node.grid == 127] = 255
        display_grid[node.grid == 0] = 220
        display_grid[node.grid == 100] = 0
        cv2.imwrite("/robot/persist/t1_project/rt_occupancy.png", display_grid)
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
