#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from phasespace_msgs.msg import Rigid
import math

# === 网格 -> 实世界 映射参数 ===
GRID_SIZE = 40
REAL_MIN = 0  # mm
REAL_MAX = 1600.0   # mm
SPARSE_DIST_MM = 50.0  # 稀疏化最小点间距

def grid_to_world(gx: float, gy: float):
    """将网格坐标(0..39)映射到真实坐标(mm)，范围 [-1500,1500]"""
    scale_x = (REAL_MAX - REAL_MIN) / (GRID_SIZE - 1)
    scale_y = (REAL_MAX - REAL_MIN) / (GRID_SIZE - 1)
    x_mm = REAL_MIN + gx * scale_x
    y_mm = REAL_MIN + gy * scale_y
    return x_mm, y_mm

# === 原始路径点 (网格坐标) ===
raw_path = [
(3, 5), (7, 12), (12, 8), (15, 3), (10, 18),
 (5, 25), (8, 30), (16, 20), (13, 28), (6, 35),
 (2, 15), (11, 22), (14, 10), (9, 32), (4, 18),
 (7, 27), (12, 35), (17, 14), (15, 25), (18, 7),
 ]

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower_direct_path')

        # === 控制参数 ===
        self.ROTATE_SPEED = 0.3
        self.ANGLE_TOLERANCE = 5.0     # deg
        self.MIN_ANGLE_THRESHOLD = 1.0 # deg
        self.DIRECTION_CORRECTION = -1
        self.LINEAR_SPEED = 0.15
        self.DIST_THRESHOLD_MM = 60.0
        self.SKIP_THRESHOLD_MM = 40.0

        self.cmd_pub = self.create_publisher(Twist, '/tb3_4/cmd_vel', 10)
        self.create_subscription(Rigid, '/phasespace_body_robot4', self.control_callback, 10)

        # === 稀疏化路径 (先映射到 world 坐标再稀疏化) ===
        self.path_grid = []
        self.path_world = []
        last_x, last_y = None, None
        for gx, gy in raw_path:
            x_mm, y_mm = grid_to_world(gx, gy)
            if last_x is None or math.hypot(x_mm - last_x, y_mm - last_y) >= SPARSE_DIST_MM:
                self.path_grid.append((gx, gy))
                self.path_world.append((x_mm, y_mm))
                last_x, last_y = x_mm, y_mm

        # === 打印稀疏化后的路径点 ===
        self.get_logger().info(f"🛣 稀疏化后的路径点 ({len(self.path_world)} 点):")
        for i, ((gx, gy), (x, y)) in enumerate(zip(self.path_grid, self.path_world), 1):
            self.get_logger().info(
                f"  {i:03d}: grid=({gx:.2f},{gy:.2f}) -> world=({x:.1f},{y:.1f}) mm"
            )

        self.index = 0
        self.current_heading = 0.0
        self.aligned = False
        self.target_x_mm = None
        self.target_y_mm = None
        self.reached = False

        if len(self.path_world) > 0:
            self.target_x_mm, self.target_y_mm = self.path_world[0]
            self.get_logger().info(f"🎯 初始目标: ({self.target_x_mm:.1f}, {self.target_y_mm:.1f}) 共 {len(self.path_world)} 点")
        else:
            self.get_logger().warn("⚠️ 路径为空！")

    def advance_target(self, msg: Rigid):
        self.index += 1
        while self.index < len(self.path_world):
            nx, ny = self.path_world[self.index]
            if math.hypot(nx - msg.x, ny - msg.z) < self.SKIP_THRESHOLD_MM:
                self.get_logger().info(f"⏭️ 跳过过近点: ({nx:.1f}, {ny:.1f})")
                self.index += 1
            else:
                self.target_x_mm, self.target_y_mm = nx, ny
                self.aligned = False
                self.reached = False
                self.get_logger().info(f"📍 新目标: ({nx:.1f}, {ny:.1f})")
                return

        self.get_logger().info("✅ 所有目标完成，停车")
        self.cmd_pub.publish(Twist())
        self.target_x_mm = None
        self.target_y_mm = None



    def control_callback(self, msg: Rigid):
        if self.target_x_mm is None or self.index >= len(self.path_world):
            return

        # 当前朝向
        siny_cosp = 2.0 * (msg.qw * msg.qy + msg.qx * msg.qz)
        cosy_cosp = 1.0 - 2.0 * (msg.qy**2 + msg.qz**2)
        # 修改为新的 heading 计算方式
        self.current_heading =  (-math.degrees(math.atan2(siny_cosp, cosy_cosp))-70) % 360

        dx = self.target_x_mm - msg.x
        dy = self.target_y_mm - msg.z
        dist_mm = math.hypot(dx, dy)

        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (target_angle - self.current_heading + 180) % 360 - 180

        twist = Twist()

        if dist_mm <= self.DIST_THRESHOLD_MM:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if not self.reached:
                self.reached = True
                self.get_logger().info(f"📌 到达目标点: ({self.target_x_mm:.1f}, {self.target_y_mm:.1f})")
            self.advance_target(msg)
        else:
            # === 对齐阶段 ===
            if not self.aligned:
                if abs(angle_diff) > max(self.ANGLE_TOLERANCE, self.MIN_ANGLE_THRESHOLD):
                    twist.angular.z = self.DIRECTION_CORRECTION * math.copysign(self.ROTATE_SPEED, angle_diff)
                    twist.linear.x = 0.0
                else:
                    self.aligned = True
            # === 前进阶段 ===
            if self.aligned:
                twist.linear.x = self.LINEAR_SPEED
                twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


