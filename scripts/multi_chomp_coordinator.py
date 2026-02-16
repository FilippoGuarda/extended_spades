#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from extended_spades.action import MultiChompOptimize
from rclpy.executors import MultiThreadedExecutor


class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        # Configuration
        self.declare_parameter('robot_count', 6)
        self.declare_parameter('controller_id', 'FollowPath')
        
        self.robot_count = self.get_parameter('robot_count').value
        self.controller_id = self.get_parameter('controller_id').value
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        
        self.get_logger().info(f"Coordinating {self.robot_count} robots: {self.robot_names}")
        
        self.goals = {}
        self.raw_paths = {}
        self.optimization_in_progress = False
        self.nav2_plan_clients = {}
        self.nav2_exec_clients = {}

        # Initialize action clients
        for name in self.robot_names:
            plan_topic = f'/{name}/compute_path_to_pose'
            exec_topic = f'/{name}/follow_path'
            
            self.get_logger().info(f"Creating clients for {name}:")
            self.get_logger().info(f"  - Planner: {plan_topic}")
            self.get_logger().info(f"  - Controller: {exec_topic}")
            
            self.nav2_plan_clients[name] = ActionClient(self, ComputePathToPose, plan_topic)
            self.nav2_exec_clients[name] = ActionClient(self, FollowPath, exec_topic)
        
        # CHOMP optimizer client
        self.chomp_client = ActionClient(self, MultiChompOptimize, 'multi_chomp_optimize')
        
        # Goal subscribers
        self.goal_subs = []
        for name in self.robot_names:
            topic = f'/{name}/spades_goal'
            self.goal_subs.append(
                self.create_subscription(
                    PoseStamped, 
                    topic, 
                    lambda msg, n=name: self.goal_callback(msg, n), 
                    10
                )
            )
        
        # Timers
        self.create_timer(0.5, self.coordination_loop)
        self.check_timer = self.create_timer(2.0, self.check_action_servers_once)

    def check_action_servers(self):
        """Verify all action servers are available"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("CHECKING ACTION SERVER AVAILABILITY")
        self.get_logger().info("=" * 60)
        
        # Check planner clients
        for name, client in self.nav2_plan_clients.items():
            ready = client.wait_for_server(timeout_sec=1.0)
            status = "✓ READY" if ready else "✗ NOT READY"
            self.get_logger().info(f"  Planner {name}: {status}")
        
        # Check controller clients
        for name, client in self.nav2_exec_clients.items():
            ready = client.wait_for_server(timeout_sec=1.0)
            status = "✓ READY" if ready else "✗ NOT READY"
            self.get_logger().info(f"  Controller {name}: {status}")
        
        # Check CHOMP
        ready = self.chomp_client.wait_for_server(timeout_sec=1.0)
        status = "✓ READY" if ready else "✗ NOT READY"
        self.get_logger().info(f"  CHOMP optimizer: {status}")
        self.get_logger().info("=" * 60)

    def check_action_servers_once(self):
        """Run check_action_servers once and cancel timer"""
        self.check_action_servers()
        self.check_timer.cancel()

    def goal_callback(self, msg, robot_name):
        self.get_logger().info(
            f"Received goal for {robot_name}: "
            f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        self.goals[robot_name] = msg
        
        # Clear existing path to trigger replanning
        if robot_name in self.raw_paths:
            del self.raw_paths[robot_name]

    def coordination_loop(self):
        # Skip if CHOMP not ready or optimization in progress
        if not self.chomp_client.server_is_ready():
            return
        
        if self.optimization_in_progress:
            return

        # Request paths for robots with goals but no path
        for name, client in self.nav2_plan_clients.items():
            if name in self.goals and name not in self.raw_paths:
                if not client.server_is_ready():
                    continue

                self.get_logger().info(f"Requesting path for {name}")
                goal_msg = ComputePathToPose.Goal()
                goal_msg.goal = self.goals[name]
                goal_msg.planner_id = "GridBased"
                
                future = client.send_goal_async(goal_msg)
                future.add_done_callback(
                    lambda f, n=name: self.nav2_plan_response_callback(f, n)
                )

        # Trigger optimization when all active robots have paths
        active_robots = [r for r in self.robot_names if r in self.goals]
        
        if len(active_robots) > 0 and all(r in self.raw_paths for r in active_robots):
            self.trigger_optimization(active_robots)

    def nav2_plan_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f"Nav2 planning rejected for {robot_name}")
                return
            
            self.get_logger().info(f"Planning accepted for {robot_name}, waiting for result...")
            res_future = goal_handle.get_result_async()
            res_future.add_done_callback(
                lambda f, n=robot_name: self.nav2_plan_result_callback(f, n)
            )
        except Exception as e:
            self.get_logger().error(f"Plan request failed for {robot_name}: {e}")

    def nav2_plan_result_callback(self, future, robot_name):
        try:
            result = future.result().result
            path = result.path
            
            if len(path.poses) > 0:
                self.get_logger().info(
                    f"Received path for {robot_name} with {len(path.poses)} poses"
                )
                self.raw_paths[robot_name] = path
            else:
                self.get_logger().warn(f"Empty path received for {robot_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to get plan result for {robot_name}: {e}")

    def trigger_optimization(self, active_robots):
        if self.optimization_in_progress:
            return
        
        self.optimization_in_progress = True
        
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = len(active_robots)
        goal_msg.max_iterations = 100
        
        for name in active_robots:
            goal_msg.input_paths.append(self.raw_paths[name])

        self.get_logger().info(f"=" * 60)
        self.get_logger().info(f"TRIGGERING OPTIMIZATION FOR {len(active_robots)} ROBOTS")
        self.get_logger().info(f"=" * 60)
        
        future = self.chomp_client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda f, robots=list(active_robots): self.optimization_response_callback(f, robots)
        )

    def optimization_response_callback(self, future, active_robots):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('CHOMP optimization REJECTED!')
                self.optimization_in_progress = False
                return
            
            self.get_logger().info('Optimization ACCEPTED, waiting for result...')
            res_future = goal_handle.get_result_async()
            res_future.add_done_callback(
                lambda f, robots=active_robots: self.optimization_result_callback(f, robots)
            )
        except Exception as e:
            self.get_logger().error(f'Optimization request failed: {e}')
            self.optimization_in_progress = False

    def optimization_result_callback(self, future, active_robots):
        try:
            result = future.result().result
            paths = result.optimized_paths
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(
                f"OPTIMIZATION COMPLETE: {len(paths)} paths for {len(active_robots)} robots"
            )
            self.get_logger().info("=" * 60)
            
            if len(paths) != len(active_robots):
                self.get_logger().error(
                    f"Path count mismatch: got {len(paths)}, expected {len(active_robots)}"
                )
                self.optimization_in_progress = False
                return

            # Execute optimized paths
            for i, robot_name in enumerate(active_robots):
                opt_path = paths[i]
                
                if len(opt_path.poses) == 0:
                    self.get_logger().warn(f"Empty optimized path for {robot_name}, skipping")
                    continue
                
                self.get_logger().info(
                    f"Executing path for {robot_name} ({len(opt_path.poses)} poses)"
                )
                self.execute_path(robot_name, opt_path)
            
            # Clear processed goals and paths
            for name in active_robots:
                if name in self.goals:
                    del self.goals[name]
                if name in self.raw_paths:
                    del self.raw_paths[name]
            
            self.optimization_in_progress = False
            self.get_logger().info("All paths sent to controllers")
            
        except Exception as e:
            self.get_logger().error(f'Optimization result callback failed: {e}')
            self.optimization_in_progress = False

    def execute_path(self, robot_name, path):
        """Send optimized path to robot's controller"""
        client = self.nav2_exec_clients.get(robot_name)
        
        if not client:
            self.get_logger().error(f"No execution client for {robot_name}")
            return
        
        # Wait for controller server with timeout
        self.get_logger().info(f"Waiting for controller server for {robot_name}...")
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"Controller server for {robot_name} not available after 5s timeout!"
            )
            self.get_logger().error(
                f"Expected action server at: /{robot_name}/follow_path"
            )
            return
        
        self.get_logger().info(f"Controller server for {robot_name} is ready")
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = self.controller_id
        
        self.get_logger().info(
            f"Sending FollowPath goal to {robot_name} with controller '{self.controller_id}'"
        )
        
        future = client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda f, n=robot_name: self.execute_response_callback(f, n)
        )

    def execute_response_callback(self, future, robot_name):
        """Handle path execution response"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info(f"✓ Controller ACCEPTED path for {robot_name}")
                
                # Monitor result
                res_future = goal_handle.get_result_async()
                res_future.add_done_callback(
                    lambda f, n=robot_name: self.execute_result_callback(f, n)
                )
            else:
                self.get_logger().error(f"✗ Controller REJECTED path for {robot_name}")
        except Exception as e:
            self.get_logger().error(f"Execute request failed for {robot_name}: {e}")

    def execute_result_callback(self, future, robot_name):
        """Handle path execution completion"""
        try:
            result = future.result()
            self.get_logger().info(f"✓ Path execution completed for {robot_name}")
        except Exception as e:
            self.get_logger().error(f"✗ Path execution failed for {robot_name}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    
    # Use MultiThreadedExecutor for handling multiple action callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        node.get_logger().info("Fleet Coordinator started, spinning...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Fleet Coordinator...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
