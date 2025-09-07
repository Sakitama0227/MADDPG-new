#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from phasespace_msgs.msg import Rigid

# === Grid -> World Mapping Parameters ===
GRID_SIZE = 40
REAL_MIN = 0.0      # mm
REAL_MAX = 1600.0   # mm
SPARSE_DIST_MM = 50.0  # Minimum spacing for sparse points

def grid_to_world(gx: float, gy: float):
    """Map grid coordinates (0..39) to world coordinates (mm), range [0,1600]"""
    scale_x = (REAL_MAX - REAL_MIN) / (GRID_SIZE - 1)
    scale_y = (REAL_MAX - REAL_MIN) / (GRID_SIZE - 1)
    x_mm = REAL_MIN + gx * scale_x
    y_mm = REAL_MIN + gy * scale_y
    return x_mm, y_mm

# === Raw path points (grid coordinates) ===
raw_path = [
    (22, 6), (25, 12), (30, 8), (35, 15), (28, 20),
    (33, 10), (37, 18), (24, 28), (29, 32), (36, 25),
    (21, 14), (27, 22), (32, 30), (38, 12), (26, 35),
    (31, 18), (34, 27), (23, 34), (28, 36), (36, 38),
]

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower_tb3_3')

        # === Control Parameters ===
        self.ROTATE_SPEED = 0.3
        self.ANGLE_TOLERANCE = 5.0     # deg
        self.MIN_ANGLE_THRESHOLD = 1.0 # deg
        self.DIRECTION_CORRECTION = -1
        self.LINEAR_SPEED = 0.15
        self.DIST_THRESHOLD_MM = 60.0
        self.SKIP_THRESHOLD_MM = 40.0

        # ✅ Use robot3 control topic
        self.cmd_pub = self.create_publisher(Twist, '/tb3_3/cmd_vel', 10)
        # ✅ Use robot3 localization topic
        self.create_subscription(Rigid, '/phasespace_body_robot3', self.control_callback, 10)

        # === Sparse path (map to world coordinates first, then sparsify) ===
        self.path_grid = []
        self.path_world = []
        last_x, last_y = None, None
        for gx, gy in raw_path:
            x_mm, y_mm = grid_to_world(gx, gy)
            if last_x is None or math.hypot(x_mm - last_x, y_mm - last_y) >= SPARSE_DIST_MM:
                self.path_grid.append((gx, gy))
                self.path_world.append((x_mm, y_mm))
                last_x, last_y = x_mm, y_mm

        self.get_logger().info(f"[tb3_3] 🛣 Sparse path points ({len(self.path_world)} points):")
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
            self.get_logger().info(f"[tb3_3] 🎯 Initial target: ({self.target_x_mm:.1f}, {self.target_y_mm:.1f}) Total {len(self.path_world)} points")
        else:
            self.get_logger().warn("[tb3_3] ⚠️ Path is empty!")

    def advance_target(self, msg: Rigid):
        self.index += 1
        while self.index < len(self.path_world):
            nx, ny = self.path_world[self.index]
            if math.hypot(nx - msg.x, ny - msg.z) < self.SKIP_THRESHOLD_MM:
                self.get_logger().info(f"[tb3_3] ⏭️ Skipping too-close point: ({nx:.1f}, {ny:.1f})")
                self.index += 1
            else:
                self.target_x_mm, self.target_y_mm = nx, ny
                self.aligned = False
                self.reached = False
                self.get_logger().info(f"[tb3_3] 📍 New target: ({nx:.1f}, {ny:.1f})")
                return

        self.get_logger().info("[tb3_3] ✅ All targets reached, stopping")
        self.cmd_pub.publish(Twist())
        self.target_x_mm = None
        self.target_y_mm = None

    def control_callback(self, msg: Rigid):
        if self.target_x_mm is None or self.index >= len(self.path_world):
            return

        # Current heading
        siny_cosp = 2.0 * (msg.qw * msg.qy + msg.qx * msg.qz)
        cosy_cosp = 1.0 - 2.0 * (msg.qy**2 + msg.qz**2)
        self.current_heading = (-math.degrees(math.atan2(siny_cosp, cosy_cosp)) + 7) % 360

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
                self.get_logger().info(f"[tb3_3] 📌 Reached target point: ({self.target_x_mm:.1f}, {self.target_y_mm:.1f})")
            self.advance_target(msg)
        else:
            if not self.aligned:
                if abs(angle_diff) > max(self.ANGLE_TOLERANCE, self.MIN_ANGLE_THRESHOLD):
                    twist.angular.z = self.DIRECTION_CORRECTION * math.copysign(self.ROTATE_SPEED, angle_diff)
                    twist.linear.x = 0.0
                else:
                    self.aligned = True
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
