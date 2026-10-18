cat << 'EOF' > /robot/persist/t1_project/rt_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cuvslam
import time
import cv2

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
    return T, q, t

class RealtimeNode(Node):
    def __init__(self):
        super().__init__('t1_visual_slam_node')
        
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
        
        self.grid_size, self.res = 400, 0.05
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        self.origin_offset = self.grid_size // 2
        
        self.last_l, self.last_r = None, None
        self.latest_tof_pts = None
        self.frame_count = 0
        self.start_time = time.time()
        
        self.last_valid_pos = None 
        self.world_offset = np.zeros(3)
        self.hard_limit = 0.35  
        self.latest_odom_3d = None
        self.last_odom_3d = None

        self.sub_l = self.create_subscription(Image, '/aima/hal/camera/head_stereo_left_mono', self.left_cb, 10)
        self.sub_r = self.create_subscription(Image, '/aima/hal/camera/head_stereo_right_mono', self.right_cb, 10)
        self.sub_tof = self.create_subscription(PointCloud2, '/percep/tof/front/point_cloud', self.tof_cb, 10)
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_cb, qos_be)

        self.pub_odom = self.create_publisher(Odometry, '/rt_odometry', 10)
        self.pub_grid = self.create_publisher(OccupancyGrid, '/rt_occupancy_grid', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("T1 视觉建图 ROS 2 Node 已启动！")

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.latest_odom_3d = np.array([p.x, p.y, p.z])

    def tof_cb(self, msg):
        self.latest_tof_pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)

    def left_cb(self, msg):
        self.last_l = msg
        self.process()

    def right_cb(self, msg):
        self.last_r = msg
        self.process()

    def process(self):
        if self.last_l is None or self.last_r is None:
            return
        
        msg_l, msg_r = self.last_l, self.last_r
        t_l = msg_l.header.stamp.sec + msg_l.header.stamp.nanosec * 1e-9
        t_r = msg_r.header.stamp.sec + msg_r.header.stamp.nanosec * 1e-9
        if abs(t_l - t_r) > 0.005:
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
            self.tracker = cuvslam.Tracker(self.rig, self.cfg)
            return

        ret = get_world_pose(pose_est)
        if ret is None:
            return
        T_wc, q, t = ret
        current_pos = np.array(t) + self.world_offset

        if self.last_valid_pos is not None:
            jump_dist = np.linalg.norm(current_pos - self.last_valid_pos)
            if jump_dist > self.hard_limit:
                odom_moved = np.zeros(3)
                if self.latest_odom_3d is not None and self.last_odom_3d is not None:
                    odom_moved = self.latest_odom_3d - self.last_odom_3d
                
                self.world_offset = self.last_valid_pos + odom_moved - np.array(t)
                current_pos = self.last_valid_pos + odom_moved
                
                self.tracker = cuvslam.Tracker(self.rig, self.cfg)
                self.last_valid_pos = None
                if self.latest_odom_3d is not None:
                    self.last_odom_3d = self.latest_odom_3d.copy()
                return 
                
        self.last_valid_pos = current_pos.copy()
        T_wc[:3, 3] = current_pos

        if self.latest_odom_3d is not None:
            self.last_odom_3d = self.latest_odom_3d.copy()

        tft = TransformStamped()
        tft.header.stamp = msg_l.header.stamp
        tft.header.frame_id = 'map'
        tft.child_frame_id = 'camera_link'
        tft.transform.translation.x = float(current_pos[0])
        tft.transform.translation.y = float(current_pos[2])
        tft.transform.translation.z = float(current_pos[1]) 
        tft.transform.rotation.x = float(q[1])
        tft.transform.rotation.y = float(q[3]) 
        tft.transform.rotation.z = float(-q[2])
        tft.transform.rotation.w = float(q[0])
        self.tf_broadcaster.sendTransform(tft)

        odom = Odometry()
        odom.header.stamp = msg_l.header.stamp
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'camera_link'
        odom.pose.pose.position.x = float(current_pos[0])
        odom.pose.pose.position.y = float(current_pos[2])
        odom.pose.pose.position.z = float(current_pos[1])
        odom.pose.pose.orientation.x = float(q[1])
        odom.pose.pose.orientation.y = float(q[3])
        odom.pose.pose.orientation.z = float(-q[2])
        odom.pose.pose.orientation.w = float(q[0])
        self.pub_odom.publish(odom)

        if self.latest_tof_pts is not None and self.frame_count % 2 == 0:
            pts_cam = self.latest_tof_pts
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

            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = msg_l.header.stamp
            grid_msg.header.frame_id = 'map'
            grid_msg.info.resolution = self.res
            grid_msg.info.width = self.grid_size
            grid_msg.info.height = self.grid_size
            grid_msg.info.origin.position.x = -self.origin_offset * self.res
            grid_msg.info.origin.position.y = -self.origin_offset * self.res
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0
            
            flat_grid = self.grid.flatten()
            flat_grid = np.where(flat_grid == 127, -1, flat_grid)
            grid_msg.data = flat_grid.tolist()
            
            self.pub_grid.publish(grid_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RealtimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
EOF
