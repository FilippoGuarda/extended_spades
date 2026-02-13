#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from extended_spades.action import MultiChompOptimize
import threading
import time

class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        # Configuration
        self.robot_names = ['robot1', 'robot2'] # Update with your namespaces
        self.goals = {} # Store target poses {robot_name: PoseStamped}
        self.raw_paths = {} # Store Nav2 paths {robot_name: Path}
        self.optimized_paths = {}
        
        # Clients for Nav2 (one per robot)
        self.nav2_clients = {}
        for name in self.robot_names:
            # Connect to Nav2's path planning action server
            topic = f'/{name}/compute_path_to_pose'
            client = ActionClient(self, ComputePathToPose, topic)
            self.nav2_clients[name] = client
            self.get_logger().info(f'Waiting for Nav2 server: {topic}')
            # client.wait_for_server() # Uncomment to block until ready

        # Client for Multi-CHOMP
        self.chomp_client = ActionClient(self, MultiChompOptimize, 'multi_chomp_optimize')
        self.get_logger().info('Waiting for Multi-CHOMP server...')
        self.chomp_client.wait_for_server()

        # Timer to coordinate the loop (e.g., 2Hz)
        self.create_timer(0.5, self.coordination_loop)
        
        # Test: Set dummy goals for start (Replace with real goal subscription later)
        self.set_dummy_goals()

    def set_dummy_goals(self):
        # Example: Crossing paths
        # Robot 1: (0,0) -> (5,5)
        goal1 = PoseStamped()
        goal1.header.frame_id = 'map'
        goal1.pose.position.x = 5.0
        goal1.pose.position.y = 5.0
        self.goals['robot1'] = goal1

        # Robot 2: (5,0) -> (0,5)
        goal2 = PoseStamped()
        goal2.header.frame_id = 'map'
        goal2.pose.position.x = 0.0
        goal2.pose.position.y = 5.0
        self.goals['robot2'] = goal2

    def coordination_loop(self):
        """Main loop: Get Nav2 paths -> Bundle -> Optimize"""
        
        # 1. Request Paths from Nav2 for all robots
        futures = []
        for name, client in self.nav2_clients.items():
            if name in self.goals:
                goal_msg = ComputePathToPose.Goal()
                goal_msg.goal = self.goals[name]
                goal_msg.planner_id = "GridBased" # Or your preferred planner
                
                # Send async request
                future = client.send_goal_async(goal_msg)
                futures.append((name, future))

        # 2. Wait for all paths (Synchronous for simplicity here, but non-blocking is better for production)
        # Note: In a real system, you'd use callbacks. This is a simplified logic.
        updated_any = False
        
        for name, future in futures:
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            if future.result():
                goal_handle = future.result()
                res_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, res_future, timeout_sec=1.0)
                
                if res_future.result():
                    path = res_future.result().result.path
                    if len(path.poses) > 0:
                        self.raw_paths[name] = path
                        updated_any = True

        # 3. If we have paths for all robots, trigger optimization
        if updated_any and len(self.raw_paths) == len(self.robot_names):
            self.trigger_optimization()

    def trigger_optimization(self):
        self.get_logger().info("Triggering Multi-CHOMP optimization...")
        
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = len(self.robot_names)
        goal_msg.max_iterations = 100
        
        # Ensure order matches self.robot_names list
        for name in self.robot_names:
            goal_msg.input_paths.append(self.raw_paths[name])

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
        
        self.get_logger().info(f"Received {len(paths)} optimized paths")
        
        # 4. Publish or distribute updated paths
        # Here you would typically publish these to a path following controller
        # e.g., /robot1/controller/follow_path
        
        for i, name in enumerate(self.robot_names):
            if i < len(paths):
                # Example: publish to visualization or controller
                pass

def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()