#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs2.action import ComputePathToPose
from extended_spades.action import MultiChompOptimize

class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        self.declare_parameter('robot_count', 6)
        self.robot_count = self.get_parameter('robot_count').value
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        self.get_logger().info(f"Coordinating {self.robot_count} robots: {self.robot_names}")
        self.goals = {}
        self.raw_paths = {}
        self.nav2_clients = {}
        for name in self.robot_names:
            topic = f'/{name}/compute_path_to_pose'
            client = ActionClient(self, ComputePathToPose, topic)
            self.nav2_clients[name] = client
        self.chomp_client = ActionClient(self, MultiChompOptimize, 'multi_chomp_optimize')
        self.create_timer(0.5, self.coordination_loop)
        self.goal_subs = []
        for name in self.robot_names:
            topic = f'/{name}/goal_pose'
            self.goal_subs.append(
                self.create_subscription(
                    PoseStamped, 
                    topic, 
                    lambda msg, n=name: self.goal_callback(msg, n), 
                    10
                )
            )

    def goal_callback(self, msg, robot_name):
        self.get_logger().info(f"Received goal for {robot_name}")
        self.goals[robot_name] = msg
        if robot_name in self.raw_paths:
            del self.raw_paths[robot_name]

    def coordination_loop(self):
        if not self.chomp_client.server_is_ready():
            if self.chomp_client.wait_for_server(timeout_sec=0.1):
                pass
            else:
                return

        # request paths from nav2
        for name, client in self.nav2_clients.items():
            if name in self.goals and name not in self.raw_paths:
                if not client.server_is_ready():
                    continue 

                goal_msg = ComputePathToPose.Goal()
                goal_msg.goal = self.goals[name]
                goal_msg.planner_id = "GridBased" 
                
                future = client.send_goal_async(goal_msg)
                future.add_done_callback(lambda f, n=name: self.nav2_goal_response_callback(f, n))

        active_robots = [r for r in self.robot_names if r in self.goals]
        
        if len(active_robots) > 0 and all(r in self.raw_paths for r in active_robots):
            self.trigger_optimization(active_robots)

    def nav2_goal_response_callback(self, future, robot_name):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Nav2 rejected goal for {robot_name}")
            return
        
        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(lambda f, n=robot_name: self.nav2_result_callback(f, n))

    def nav2_result_callback(self, future, robot_name):
        result = future.result().result
        path = result.path
        if len(path.poses) > 0:
            self.raw_paths[robot_name] = path
            # self.get_logger().info(f"Got path for {robot_name}")

    def trigger_optimization(self, active_robots):
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = len(active_robots)
        goal_msg.max_iterations = 100
        
        for name in active_robots:
            goal_msg.input_paths.append(self.raw_paths[name])

        self.get_logger().info(f"Optimizing for {len(active_robots)} robots...")

        self.raw_paths.clear() 

        future = self.chomp_client.send_goal_async(goal_msg)
        future.add_done_callback(self.optimization_response_callback)

    def optimization_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Optimization rejected')
            return
        res_future = goal_handle.get_result_async()
        res_future.add_done_callback(self.optimization_result_callback)

    def optimization_result_callback(self, future):
        result = future.result().result
        paths = result.optimized_paths
        self.get_logger().info(f"Optimization complete. Received {len(paths)} paths.")
        # TODO: publish optimized paths to /robotX/plan or /robotX/path_controller...

def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
