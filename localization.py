#!/usr/bin/env python3
import math
import threading
import time
import rclpy
from rclpy.node import Node
from phasespace_msgs.msg import Rigid

_localization_node = None
_spin_thread = None

class MultiAgentLocalization(Node):
    def __init__(self, agent_topics):
        super().__init__('multi_agent_localization')

        self.real_min_x = 0.0
        self.real_min_y = 0.0
        self.real_max_x = 1600.0
        self.real_max_y = 1600.0
        self.grid_size = 40

        self._lock = threading.Lock()
        self._data = {agent_id: None for agent_id in agent_topics}

        for agent_id, topic in agent_topics.items():
            self.create_subscription(
                Rigid,
                topic,
                lambda msg, aid=agent_id: self.rigid_callback(msg, aid),
                10
            )

    def real_to_grid(self, real_x, real_y):
        scale_x = (self.grid_size - 1) / (self.real_max_x - self.real_min_x)
        scale_y = (self.grid_size - 1) / (self.real_max_y - self.real_min_y)
        grid_x = int(round((real_x - self.real_min_x) * scale_x))
        grid_y = int(round((real_y - self.real_min_y) * scale_y))
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        return grid_x, grid_y

    def rigid_callback(self, msg, agent_id):
        # phasespace: y -> z
        real_x = msg.x / 1000.0  # mm -> m
        real_y = msg.z / 1000.0
        siny_cosp = 2.0 * (msg.qw * msg.qy + msg.qx * msg.qz)
        cosy_cosp = 1.0 - 2.0 * (msg.qy**2 + msg.qz**2)
        heading = (-math.degrees(math.atan2(siny_cosp, cosy_cosp)) - 70) % 360
        with self._lock:
            self._data[agent_id] = (real_x, real_y, heading)

    def get_agent_position(self, agent_id):
        with self._lock:
            if self._data[agent_id] is None:
                return None
            real_x, real_y, heading = self._data[agent_id]
            grid_x, grid_y = self.real_to_grid(real_x * 1000, real_y * 1000)
            return real_x, real_y, heading, grid_x, grid_y


def start_localization_node():
    global _localization_node, _spin_thread
    if _localization_node is not None:
        return _localization_node

    if not rclpy.ok():
        rclpy.init()

    agent_topics = {
        0: '/phasespace_body_robot4',
        1: '/phasespace_body_robot3'
    }
    _localization_node = MultiAgentLocalization(agent_topics)

    # Start spin thread
    _spin_thread = threading.Thread(target=rclpy.spin, args=(_localization_node,), daemon=True)
    _spin_thread.start()
    time.sleep(0.5)  # Wait for node to stabilize
    return _localization_node


def get_current_position(agent_id):
    node = start_localization_node()
    return node.get_agent_position(agent_id)


if __name__ == "__main__":
    loc_node = start_localization_node()
    try:
        while True:
            for aid in [0, 1]:
                pos = loc_node.get_agent_position(aid)
                if pos:
                    rx, ry, hd, gx, gy = pos
                    print(f"Agent {aid} Real Position: ({rx:.2f},{ry:.2f}) m | Grid: ({gx},{gy}) | Heading: {hd:.2f}°")
                else:
                    print(f"Agent {aid} No localization data yet")
            time.sleep(0.5)
    except KeyboardInterrupt:
        rclpy.shutdown()
