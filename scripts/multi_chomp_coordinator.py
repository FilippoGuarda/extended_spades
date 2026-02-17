#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import ComputePathToPose, FollowPath
from extended_spades.action import MultiChompOptimize
from action_msgs.msg import GoalStatus
import nav_msgs.msg
import tf2_ros
from tf2_ros import Buffer, TransformListener
import math

class FleetCoordinator(Node):
    def __init__(self):
        super().__init__('fleet_coordinator')

        # --- Configuration ---
        self.declare_parameter('robot_count', 6)
        self.declare_parameter('controller_id', 'FollowPath')
        
        self.robot_count = self.get_parameter('robot_count').value
        self.controller_id = self.get_parameter('controller_id').value
        self.robot_names = [f'robot{i}' for i in range(1, self.robot_count + 1)]
        
        self.get_logger().info(f"Fleet Coordinator Active: {self.robot_names}")

        # --- State Management ---
        # goals: stores active goals for robots requested to move
        self.goals = {}
        # new_plan_buffer: temporary storage for plans returned by Nav2 before optimization
        self.new_plan_buffer = {} 
        self.optimization_in_progress = False
        self.pending_plan_requests = set()
        
        # --- TF Buffer (Critical for finding Robot Position) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Clients ---
        self.nav2_plan_clients = {}
        self.nav2_exec_clients = {}
        self.path_debug_pubs = {}
        for name in self.robot_names:
            self.nav2_plan_clients[name] = ActionClient(self, ComputePathToPose, f'/{name}/compute_path_to_pose')
            self.nav2_exec_clients[name] = ActionClient(self, FollowPath, f'/{name}/follow_path')
            self.path_debug_pubs[name] = self.create_publisher(
                                                nav_msgs.msg.Path, 
                                                f'/{name}/debug/chomp_optimized_path',  # Namespaced
                                                10
                                            )
        
        self.chomp_client = ActionClient(self, MultiChompOptimize, 'multi_chomp_optimize')
        
        # --- Visualization ---


        # --- Subscribers ---
        self.goal_subs = []
        for name in self.robot_names:
            self.goal_subs.append(
                self.create_subscription(
                    PoseStamped, 
                    f'/{name}/spades_goal', 
                    lambda msg, n=name: self.goal_callback(msg, n), 
                    10
                )
            )
        
        # --- Loop ---
        self.create_timer(0.5, self.coordination_loop)

    def get_robot_pose(self, robot_name):
        """Get the current pose of the robot in the map frame."""
        try:
            target_frame = 'map'
            source_frame = f'{robot_name}/base_link'
            
            if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                return None

            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            
            pose = PoseStamped()
            pose.header.frame_id = target_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = t.transform.translation.x
            pose.pose.position.y = t.transform.translation.y
            pose.pose.position.z = t.transform.translation.z
            pose.pose.orientation = t.transform.rotation
            return pose

        except Exception as e:
            # Throttle log to avoid spam
            return None

    def create_holding_path(self, robot_name, length=20):
        """Generates a static path at the robot's current position."""
        pose = self.get_robot_pose(robot_name)
        if not pose:
            self.get_logger().warn(f"Cannot create holding path for {robot_name}: No TF")
            return None
        
        path = nav_msgs.msg.Path()
        path.header = pose.header
        # Fill path with identical poses
        path.poses = [pose for _ in range(length)]
        return path

    def goal_callback(self, msg, robot_name):
        self.get_logger().info(f"Goal received for {robot_name}")
        self.goals[robot_name] = msg
        # Clear any old buffered plan for this robot
        self.new_plan_buffer.pop(robot_name, None)
        if robot_name in self.pending_plan_requests:
             self.pending_plan_requests.remove(robot_name)

    def coordination_loop(self):
        if not self.chomp_client.server_is_ready() or self.optimization_in_progress:
            return

        # 1. Request Nav2 Plans ONLY for robots with NEW goals
        for name, client in self.nav2_plan_clients.items():
            if name in self.goals and name not in self.new_plan_buffer:
                if name in self.pending_plan_requests or not client.server_is_ready():
                    continue

                self.get_logger().info(f"Requesting Global Plan for {name}...")
                self.pending_plan_requests.add(name)
                
                goal_msg = ComputePathToPose.Goal()
                goal_msg.goal = self.goals[name]
                goal_msg.planner_id = "GridBased"
                goal_msg.use_start = False # Use current robot pose
                
                future = client.send_goal_async(goal_msg)
                future.add_done_callback(lambda f, n=name: self.nav2_plan_response_callback(f, n))

        # 2. Check if we are ready to optimize
        # We need plans for ALL robots that have active goals.
        # Robots without goals will just use their current position.
        
        robots_with_goals = [r for r in self.robot_names if r in self.goals]
        robots_with_plans = [r for r in robots_with_goals if r in self.new_plan_buffer]
        
        # If we have at least one active request, and all needed plans are ready:
        if len(robots_with_goals) > 0 and len(robots_with_plans) == len(robots_with_goals):
             self.trigger_fleet_optimization()

    def nav2_plan_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.pending_plan_requests.discard(robot_name)
                return
            
            goal_handle.get_result_async().add_done_callback(
                lambda f, n=robot_name: self.nav2_plan_result_callback(f, n)
            )
        except Exception:
            self.pending_plan_requests.discard(robot_name)

    def nav2_plan_result_callback(self, future, robot_name):
        try:
            result = future.result().result
            if len(result.path.poses) > 0:
                self.new_plan_buffer[robot_name] = result.path
                self.get_logger().info(f"Plan received for {robot_name}")
            else:
                self.get_logger().warn(f"Planner returned empty path for {robot_name}")
        finally:
            self.pending_plan_requests.discard(robot_name)

    def trigger_fleet_optimization(self):
        self.optimization_in_progress = True
        
        goal_msg = MultiChompOptimize.Goal()
        goal_msg.num_robots = self.robot_count # ALWAYS send full count
        goal_msg.max_iterations = 100
        
        # Construct input vector for ALL robots (index 0 to N-1)
        # This guarantees C++ index 'i' corresponds to 'robot(i+1)'
        inputs_valid = True
        
        for name in self.robot_names:
            path_to_send = None
            
            if name in self.new_plan_buffer:
                # Case A: Robot has a new Global Plan
                path_to_send = self.new_plan_buffer[name]
            else:
                # Case B: Robot is idle/holding. Send "Stationary Path"
                # This ensures CHOMP knows this robot is an obstacle!
                path_to_send = self.create_holding_path(name)
            
            if path_to_send is None:
                self.get_logger().warn(f"Skipping optimization: Could not get state for {name}")
                inputs_valid = False
                break
                
            goal_msg.input_paths.append(path_to_send)

        if not inputs_valid:
            self.optimization_in_progress = False
            return

        self.get_logger().info(f"Triggering Fleet Optimization for {self.robot_count} robots...")
        
        self.chomp_client.send_goal_async(goal_msg).add_done_callback(
            lambda f: self.optimization_response_callback(f)
        )

    def optimization_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.optimization_in_progress = False
                return
            
            goal_handle.get_result_async().add_done_callback(
                lambda f: self.optimization_result_callback(f)
            )
        except Exception as e:
            self.get_logger().error(f"Optimization request failed: {e}")
            self.optimization_in_progress = False

    def optimization_result_callback(self, future):
        try:
            result = future.result().result
            optimized_paths = result.optimized_paths
            
            if len(optimized_paths) != self.robot_count:
                self.get_logger().error("Mismatch in optimized paths count!")
                self.optimization_in_progress = False
                return

            self.get_logger().info("Optimization Complete. Distributing paths...")

            # Iterate over ALL robots
            for i, robot_name in enumerate(self.robot_names):
                opt_path = optimized_paths[i]
                
                if len(opt_path.poses) < 2:
                    continue

                # --- VALIDATION & EXECUTION LOGIC ---
                # We execute the path if:
                # 1. The robot explicitly requested a goal (it's in self.goals)
                # 2. OR: The path implies significant movement (CHOMP nudged it)
                # For safety/simplicity, we execute ALL paths that look valid.
                
                # Check for Inversion (End closer than Start)
                robot_pose = self.get_robot_pose(robot_name)
                if robot_pose:
                    rx, ry = robot_pose.pose.position.x, robot_pose.pose.position.y
                    sx, sy = opt_path.poses[0].pose.position.x, opt_path.poses[0].pose.position.y
                    ex, ey = opt_path.poses[-1].pose.position.x, opt_path.poses[-1].pose.position.y
                    
                    dist_start = math.hypot(sx - rx, sy - ry)
                    dist_end = math.hypot(ex - rx, ey - ry)
                    
                    if dist_end < dist_start and dist_end < 1.0:
                        self.get_logger().info(f"Fixing inverted path for {robot_name}")
                        opt_path.poses.reverse()
                        
                        # Recompute orientations for reversed path
                        for i in range(len(opt_path.poses)):
                            if i < len(opt_path.poses) - 1:
                                # Forward difference: direction to next pose
                                p_curr = opt_path.poses[i].pose.position
                                p_next = opt_path.poses[i + 1].pose.position
                                yaw = math.atan2(p_next.y - p_curr.y, p_next.x - p_curr.x)
                            else:
                                # Last pose: copy orientation from previous
                                yaw = math.atan2(
                                    opt_path.poses[i].pose.position.y - opt_path.poses[i-1].pose.position.y,
                                    opt_path.poses[i].pose.position.x - opt_path.poses[i-1].pose.position.x
                                )
                            
                            # Convert yaw to quaternion
                            half_yaw = yaw * 0.5
                            opt_path.poses[i].pose.orientation.x = 0.0
                            opt_path.poses[i].pose.orientation.y = 0.0
                            opt_path.poses[i].pose.orientation.z = math.sin(half_yaw)
                            opt_path.poses[i].pose.orientation.w = math.cos(half_yaw)
                    elif dist_start > 2.0:

                         self.get_logger().error(f"Path for {robot_name} detached (gap={dist_start:.2f}m). Ignoring.")
                         continue

                self.execute_path(robot_name, opt_path)
            
            # Reset active goals
            # Note: We clear goals so we don't re-request plans, 
            # but we keep monitoring.
            self.goals.clear()
            self.new_plan_buffer.clear()
            self.optimization_in_progress = False
            
        except Exception as e:
            self.get_logger().error(f"Optimization callback exception: {e}")
            self.optimization_in_progress = False

    def execute_path(self, robot_name, path):
        client = self.nav2_exec_clients.get(robot_name)
        if not client: return

        # Get current robot pose
        robot_pose = self.get_robot_pose(robot_name)
        if not robot_pose:
            self.get_logger().warn(f"Cannot execute path for {robot_name}: No TF")
            return

        # Prepend current pose to ensure MPPI can track from current location
        path.poses.insert(0, robot_pose)

        # Refresh timestamps
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        path.header.frame_id = "map"
        for pose in path.poses:
            pose.header.stamp = now
            pose.header.frame_id = "map"

        self.path_debug_pubs[robot_name].publish(path)

        if not client.wait_for_server(timeout_sec=1.0):
            return
        
        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = self.controller_id
        
        client.send_goal_async(goal_msg).add_done_callback(
            lambda f, n=robot_name: self.execute_response_callback(f, n)
        )

    def execute_response_callback(self, future, robot_name):
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                # self.get_logger().info(f"Controller started for {robot_name}")
                pass
            else:
                self.get_logger().error(f"Controller REJECTED path for {robot_name}")
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = FleetCoordinator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
