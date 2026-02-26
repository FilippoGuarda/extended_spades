#include "extended_spades/multi_chomp.hpp"

using std::placeholders::_1;

MultiChompNode::MultiChompNode() : Node("multi_chomp_server") {

    load_parameters();
    init_matrices();
    // ROS interface
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray> ("plan_markers", 20);
    rclcpp::QoS map_qos(1);
    map_qos.transient_local();

    // occupancy grid subscription
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>
        ("robot1/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

    // timer (using action call for syncing the optimization step)
    // timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&MultiChompNode::solve_step, this));

    RCLCPP_INFO(this->get_logger(), "multi chomp server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters()
{
    this->declare_parameter<int>("num_robots", 6);
    this->declare_parameter<int>("waypoints_per_robot", 20);
    this->declare_parameter<double>("dt", 0.1);
    this->declare_parameter<double>("eta", 100.0);
    this->declare_parameter<double>("lambda", 0.1);
    this->declare_parameter<double>("mu", 0.4);
    // assuming robot_radius and obstacle_max_dist are also parameters or constants
    params_.robot_radius = 0.5;
    params_.obstacle_max_dist = 4.0;
    params_.num_robots = this->get_parameter("num_robots").as_int();
    params_.waypoints_per_robot = this->get_parameter("waypoints_per_robot").as_int();
    params_.dt = this->get_parameter("dt").as_double();
    params_.eta = this->get_parameter("eta").as_double();
    params_.lambda = this->get_parameter("lambda").as_double();
    params_.mu = this->get_parameter("mu").as_double();

    xidim_ = params_.num_robots * params_.waypoints_per_robot * cdim_;
    xi_  = VectorXd::Zero(xidim_);
    bbR_ = VectorXd::Zero(xidim_);
}

void MultiChompNode::init_matrices() {

    // single robot metric A from waypoints
    size_t nq = params_.waypoints_per_robot; 
    AA_ = MatrixXd::Zero(nq*cdim_, nq*cdim_);
    
    for (size_t i=0; i < nq; ++i) { 
        AA_.block(cdim_ * i, cdim_ * i, cdim_, cdim_) = 2.0 * MatrixXd::Identity(cdim_, cdim_);

        // outside of diagonal
        if (i > 0) {
            AA_.block(cdim_ * (i - 1), cdim_ * i, cdim_, cdim_) = - 1.0 * MatrixXd::Identity(cdim_, cdim_);
            AA_.block(cdim_ * i, cdim_ * (i - 1), cdim_, cdim_) = - 1.0 * MatrixXd::Identity(cdim_, cdim_); 
        }
    }

    // now create multi robot metric, each block is one robot smoothness cost
    // robots only intaract through the interference objective (computed in solver)
    AAR_ = MatrixXd::Zero(xidim_, xidim_);
    for (int k = 0; k < params_.num_robots; ++k) {
        size_t offset = static_cast<size_t>(k) * nq * cdim_;
        AAR_.block(offset, offset, nq * cdim_, nq * cdim_) = AA_;
    }
    AAR_ /= (params_.dt * params_.dt * (nq + 1));
    AARinv_ = AAR_.inverse();
}

// map callback placeholder
void MultiChompNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {

    std::lock_guard<std::mutex> lock(map_mutex_);
    current_map_ = *msg;
    update_distance_map(*msg);
    map_received_ = true;
}

void MultiChompNode::update_distance_map(const nav_msgs::msg::OccupancyGrid& grid)
{
    map_resolution_ = grid.info.resolution;
    map_origin_x_ = grid.info.origin.position.x;
    map_origin_y_ = grid.info.origin.position.y;
    map_width_ = grid.info.width;
    map_height_ = grid.info.height;

    cv::Mat bin_img(map_height_, map_width_, CV_8UC1);

    for (int i = 0; i < map_height_; ++i) {
        for (int j = 0; j < map_width_; ++j) {
            int8_t val = grid.data[i * map_width_ + j];
            bin_img.at<uint8_t>(i, j) = (val > 50) ? 0 : 255;
        }
    }

    cv::Mat dist_img_pixels;
    cv::distanceTransform(bin_img, dist_img_pixels, cv::DIST_L2, 5);

    dist_map_ = dist_img_pixels * map_resolution_;

    cv::Sobel(dist_map_, dist_grad_x_, CV_64F, 1, 0, 3);
    cv::Sobel(dist_map_, dist_grad_y_, CV_64F, 0, 1, 3);
}

double MultiChompNode::get_environment_distance(double x, double y, Eigen::Vector2d& gradient) const
{
    if (dist_map_.empty()) {
        gradient << 0.0, 0.0;
        return 999.0;
    }

    double gx = (x - map_origin_x_) / map_resolution_;
    double gy = (y - map_origin_y_) / map_resolution_;

    if (gx < 1 || gx >= map_width_ - 1 || gy < 1 || gy >= map_height_ - 1) {
        gradient << 0.0, 0.0;
        return 999.0;
    }

    int u = static_cast<int>(gx);
    int v = static_cast<int>(gy);

    float dist = dist_map_.at<float>(v, u);
    double dx = dist_grad_x_.at<double>(v, u);
    double dy = dist_grad_y_.at<double>(v, u);

    double norm = std::sqrt(dx * dx + dy * dy);
    if (norm > 1e-5) {
        gradient << dx / norm, dy / norm;
    } else {
        gradient << 0.0, 0.0;
    }

    return static_cast<double>(dist);
}

std::vector<nav_msgs::msg::Path>
MultiChompNode::get_paths(const std::vector<nav_msgs::msg::Path> & templates) const
{
  std::vector<nav_msgs::msg::Path> out;
  out.resize(params_.num_robots);

  const int nq = params_.waypoints_per_robot;
  
  // Guard against uninitialized optimizer state
  if (xidim_ == 0 || xi_.size() != static_cast<long>(xidim_)) {
      return out; 
  }

  for (int r = 0; r < params_.num_robots; ++r) {
    nav_msgs::msg::Path path;

    // Use "map" frame and current time
    path.header.frame_id = "map"; 
    path.header.stamp = this->now();

    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    path.poses.resize(nq);

    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      Eigen::Vector2d p = xi_.block(idx, 0, cdim_, 1);

      // --- SMOOTHER ORIENTATION CALCULATION ---
      double yaw = 0.0;
      
      // Look ahead/behind to find a point far enough away to get a stable angle
      // We search up to 3 indices forward/backward
      Eigen::Vector2d p_next = p;
      Eigen::Vector2d p_prev = p;
      bool found_next = false;
      bool found_prev = false;

      // Find next distinct point
      for(int step = 1; step <= 3 && (k + step) < nq; ++step) {
          size_t idx_n = robot_offset + static_cast<size_t>(k + step) * cdim_;
          Eigen::Vector2d check = xi_.block(idx_n, 0, cdim_, 1);
          if ((check - p).squaredNorm() > 0.01) { // > 10cm squared
              p_next = check;
              found_next = true;
              break;
          }
      }

      // Find prev distinct point
      for(int step = 1; step <= 3 && (k - step) >= 0; ++step) {
          size_t idx_p = robot_offset + static_cast<size_t>(k - step) * cdim_;
          Eigen::Vector2d check = xi_.block(idx_p, 0, cdim_, 1);
          if ((check - p).squaredNorm() > 0.01) {
              p_prev = check;
              found_prev = true;
              break;
          }
      }

      if (found_next && found_prev) {
          // Central difference (smoothest)
          yaw = std::atan2(p_next.y() - p_prev.y(), p_next.x() - p_prev.x());
      } else if (found_next) {
          // Forward difference (start of path)
          yaw = std::atan2(p_next.y() - p.y(), p_next.x() - p.x());
      } else if (found_prev) {
          // Backward difference (end of path)
          yaw = std::atan2(p.y() - p_prev.y(), p.x() - p_prev.x());
      } else {
          // No movement? Keep previous yaw or 0
          if (k > 0) {
              // Copy orientation from previous pose
              const auto& prev_q = path.poses[k-1].pose.orientation;
              // Extract yaw from quat ... simpler to just keep 0 if completely still
          }
      }
      // ----------------------------------------

      // Convert Yaw to Quaternion
      double halfYaw = yaw * 0.5;
      double sz = std::sin(halfYaw);
      double cw = std::cos(halfYaw);

      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = p.x();
      ps.pose.position.y = p.y();
      ps.pose.position.z = 0.0;
      
      ps.pose.orientation.x = 0.0;
      ps.pose.orientation.y = 0.0;
      ps.pose.orientation.z = sz;
      ps.pose.orientation.w = cw;

      path.poses[k] = ps;
    }

    out[r] = std::move(path);
  }

  return out;
}




// path resampling from nav2 provided to single elements
std::vector<Eigen::Vector2d>
MultiChompNode::resample_path(const nav_msgs::msg::Path & path,
                              int num_points) const
{
  std::vector<Eigen::Vector2d> out;
  out.reserve(num_points);

  const size_t n = path.poses.size();
  if (n == 0 || num_points <= 0) {
    return out;
  }
  if (n == 1) {
    // fill with same pose if only one path received
    Eigen::Vector2d p(path.poses[0].pose.position.x,
                      path.poses[0].pose.position.y);
    out.assign(num_points, p);
    return out;
  }

  // get total arc length
  std::vector<double> s(n, 0.0);
  for (size_t i = 1; i < n; ++i) {
    const auto & p0 = path.poses[i - 1].pose.position;
    const auto & p1 = path.poses[i].pose.position;
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    s[i] = s[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  double L = s.back();
  if (L < 1e-6) {
    // arc small, approximate to single point
    Eigen::Vector2d p(path.poses[0].pose.position.x,
                      path.poses[0].pose.position.y);
    out.assign(num_points, p);
    return out;
  }

  // sample at equal spaces
  for (int k = 0; k < num_points; ++k) {
    double target_s = (static_cast<double>(k) /
                       static_cast<double>(num_points - 1)) * L;

    // find segment [i, i+1] such that s[i] <= target_s <= s[i+1]
    size_t i = 0;
    while (i + 1 < n && s[i + 1] < target_s) {
      ++i;
    }
    size_t j = std::min(i + 1, n - 1);

    double ds = s[j] - s[i];
    double alpha = (ds > 1e-6) ? (target_s - s[i]) / ds : 0.0;

    const auto & p0 = path.poses[i].pose.position;
    const auto & p1 = path.poses[j].pose.position;

    Eigen::Vector2d p;
    p.x() = (1.0 - alpha) * p0.x + alpha * p1.x;
    p.y() = (1.0 - alpha) * p0.y + alpha * p1.y;

    out.push_back(p);
  }

  return out;
}

bool MultiChompNode::set_paths(const std::vector<nav_msgs::msg::Path> & paths)
{
  int new_num_robots = static_cast<int>(paths.size());
  const int nq = params_.waypoints_per_robot;
  if (nq <= 1) {
    RCLCPP_ERROR(this->get_logger(), "set_paths: waypoints_per_robot must be > 1");
    return false;
  }

  if (new_num_robots != params_.num_robots) {
    RCLCPP_INFO(this->get_logger(), "Resizing optimizer state for %d robots", new_num_robots);
    params_.num_robots = new_num_robots;
    
    xidim_ = params_.num_robots * nq * cdim_;
    xi_    = VectorXd::Zero(xidim_);
    bbR_   = VectorXd::Zero(xidim_);
    
    start_states_.resize(params_.num_robots);
    goal_states_.resize(params_.num_robots);
    
    init_matrices(); 
  }

  if (start_states_.size() != static_cast<size_t>(params_.num_robots)) {
      start_states_.resize(params_.num_robots);
      goal_states_.resize(params_.num_robots);
  }

  for (int r = 0; r < params_.num_robots; ++r) {
    const auto & path = paths[r];

    if (path.poses.size() < 2) {
      RCLCPP_ERROR(this->get_logger(),
                   "set_paths: robot %d path has < 2 poses", r);
      return false;
    }

    // store start and goal from the raw path
    const auto & p_start = path.poses.front().pose.position;
    const auto & p_goal  = path.poses.back().pose.position;

    start_states_[r] = Eigen::Vector2d(p_start.x, p_start.y);
    goal_states_[r]  = Eigen::Vector2d(p_goal.x,  p_goal.y);

    // path to nq waypoints
    auto samples = resample_path(path, nq);
    if (static_cast<int>(samples.size()) != nq) {
      RCLCPP_ERROR(this->get_logger(),
                   "set_paths: resample failed for robot %d", r);
      return false;
    }

    // pack into xi_
    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      xi_.block(idx, 0, cdim_, 1) = samples[k];
    }
  }

  RCLCPP_INFO(this->get_logger(),
              "Loaded %d robot paths into optimizer state", params_.num_robots);
  return true;
}


void MultiChompNode::solve_step() {
  if (!map_received_) return;

  // 1. Recompute bbR (Bias for start/end points)
  const int nq = params_.waypoints_per_robot;
  const double dt = params_.dt;
  
  // The scaling factor must EXACTLY match the scaling applied to AAR_ in init_matrices()
  const double scaling_factor = 1.0 / (dt * dt * (nq + 1));

  bbR_ = VectorXd::Zero(xidim_);

  for (int r = 0; r < params_.num_robots; ++r) {
      size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;

      if (r < static_cast<int>(start_states_.size())) {
          // Force of the fixed start point on the first moving waypoint (index 0)
          bbR_.block(robot_offset, 0, cdim_, 1) = start_states_[r];
      }
      if (r < static_cast<int>(goal_states_.size())) {
          // Force of the fixed goal point on the last moving waypoint (index nq-1)
          size_t end_idx = robot_offset + static_cast<size_t>(nq - 1) * cdim_;
          bbR_.block(end_idx, 0, cdim_, 1) = goal_states_[r];
      }
  }
  // Apply the same scaling factor. The negative sign is because moving the waypoint 
  // away from the boundary creates a restorative force TOWARDS the boundary.
  bbR_ *= -scaling_factor;


  // 2. Smoothness Gradient (Unprojected)
  // nabla_smooth = AAR * xi + bbR
  VectorXd nabla_smooth = AAR_ * xi_ + bbR_;

  // 3. Obstacle & Interference Gradients (Projected)
  VectorXd nabla_obs = VectorXd::Zero(xidim_);
  VectorXd nabla_inter = VectorXd::Zero(xidim_);

  Eigen::Matrix2d JJ = Eigen::Matrix2d::Identity();

  // Loop over all robots and all waypoints
  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      VectorXd start_p = start_states_[r];
      VectorXd end_p = goal_states_[r];

      for (int i = 0; i < nq; ++i) {
          int idx = offset + i * cdim_;

          // Current point q_i
          VectorXd qq = xi_.block(idx, 0, cdim_, 1);

          // Velocity qd (Central Difference)
          VectorXd qd = VectorXd::Zero(cdim_);
          if (i == 0) {
              VectorXd q_next = xi_.block(idx + cdim_, 0, cdim_, 1);
              qd = (q_next - start_p) / (2.0 * dt);
          } else if (i == nq - 1) {
              VectorXd q_prev = xi_.block(idx - cdim_, 0, cdim_, 1);
              qd = (end_p - q_prev) / (2.0 * dt);
          } else {
              VectorXd q_next = xi_.block(idx + cdim_, 0, cdim_, 1);
              VectorXd q_prev = xi_.block(idx - cdim_, 0, cdim_, 1);
              qd = (q_next - q_prev) / (2.0 * dt);
          }

          double vel = qd.norm();
          if (vel < 1.0e-3) continue; // Avoid div by zero

          // Projection Matrix: P = I - v*v' / |v|^2
          VectorXd xdn = qd / vel;
          Eigen::Matrix2d prj = Eigen::Matrix2d::Identity() - xdn * xdn.transpose();

          // Acceleration approximated by smoothness gradient
          VectorXd xdd = nabla_smooth.block(idx, 0, cdim_, 1);

          // Curvature vector kappa = P * xdd / |v|^2
          VectorXd kappa = (prj * xdd) / (vel * vel);

          // --- Obstacle Cost ---
          Eigen::Vector2d grad_env;
          double dist = get_environment_distance(qq(0), qq(1), grad_env);

          if (dist < params_.obstacle_max_dist) {
              double gain = 2.0; 
              double term = (1.0 - dist / params_.obstacle_max_dist);
              double cost = gain * params_.obstacle_max_dist * pow(term, 3.0) / 3.0;

              // Gradient dC/dx = -gain * (1-d/max)^2 * grad_env
              VectorXd delta = -gain * pow(term, 2.0) * VectorXd(grad_env);

              // Projected gradient: vel * (P * delta - cost * kappa)
              nabla_obs.block(idx, 0, cdim_, 1) += vel * (prj * delta - cost * kappa);
          }

          // --- Interference Cost ---
          const double safety_dist = 2.0 * params_.robot_radius;
          const double gainR = 1.0;

          for (int r2 = 0; r2 < params_.num_robots; ++r2) {
              if (r == r2) continue;
              size_t offset2 = static_cast<size_t>(r2) * nq * cdim_;

              // Synchronous interference check
              VectorXd q2 = xi_.block(offset2 + i * cdim_, 0, cdim_, 1);
              VectorXd diff = qq - q2;
              double ddnorm = diff.norm();

              if (ddnorm < safety_dist && ddnorm > 1e-9) {
                   double termR = (1.0 - ddnorm / safety_dist);
                   double costR = gainR * safety_dist * pow(termR, 3.0) / 3.0;

                   // Gradient: pushes qq away from q2
                   VectorXd deltaR = -gainR * pow(termR, 2.0) * (diff / ddnorm);

                   nabla_inter.block(idx, 0, cdim_, 1) += vel * (prj * deltaR - costR * kappa);
              }
          }
      }
  }

  // 4. Update Step (Pure Covariant Gradient Descent)
  
  // Combine external non-linear gradients
  VectorXd external_grad = nabla_obs + params_.mu * nabla_inter;

  // Clip the external gradient to prevent extreme spikes near obstacles
  double g_norm = external_grad.norm();
  if (g_norm > 1e6) {
      external_grad *= (1e6 / g_norm);
  }

  // Precondition the external gradients by spreading them over the trajectory
  VectorXd smooth_external_update = AARinv_ * external_grad;
  
  // Calculate the analytical smoothness update
  // AARinv_ * bbR_ calculates the "unconstrained smooth curve" between start and end
  // (xi_ + AARinv_ * bbR_) represents how far the current trajectory deviates from that perfect curve
  VectorXd smooth_prior_update = xi_ + AARinv_ * bbR_;

  // Final Covariant Update Rule
  // The external gradients push the path away from obstacles.
  // The prior update pulls the path back towards the shortest smooth line.
  xi_ -= (1.0 / params_.eta) * (smooth_external_update + params_.lambda * smooth_prior_update);

  // 5. Hard Constraints (Start/Goal)
  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      if (r < static_cast<int>(start_states_.size())) 
          xi_.block(offset, 0, cdim_, 1) = start_states_[r];
      if (r < static_cast<int>(goal_states_.size())) 
          xi_.block(offset + (nq-1)*cdim_, 0, cdim_, 1) = goal_states_[r];
  }

  publish_state();
}


double MultiChompNode::compute_current_cost() const {
  if (!map_received_) {
    return 0.0;
  }

  double total_cost = 0.0;

  // compute smoothness cost
  VectorXd smooth_term = AAR_ * xi_ + bbR_;
  double smoothness_cost = 0.5 * xi_.dot(smooth_term);
  total_cost += params_.lambda * smoothness_cost;

  // compute obstacle cost
  double obstacle_cost = 0.0;
  const int num_points = static_cast<int>(xidim_ / cdim_);
  for (int i = 0; i < num_points; ++i) {
    int idx = i * static_cast<int>(cdim_);
    VectorXd pt = xi_.block(idx, 0, cdim_, 1);
    Eigen::Vector2d grad_env;
    double dist = get_environment_distance(pt(0), pt(1), grad_env);
    if (dist < params_.obstacle_max_dist) {
      double penetration = params_.obstacle_max_dist - dist;
      obstacle_cost += penetration * penetration;
    }
  }
  total_cost += obstacle_cost;

  // compute cost of robot on robot collision
  double interference_cost = 0.0;
  const double safety_dist = 2.0 * params_.robot_radius;
  for (int r1 = 0; r1 < params_.num_robots; ++r1) {
    for (int r2 = r1 + 1; r2 < params_.num_robots; ++r2) {
      for (int k = 0; k < params_.waypoints_per_robot; ++k) {
        int idx1 = (r1 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);
        int idx2 = (r2 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);
        VectorXd p1 = xi_.block(idx1, 0, cdim_, 1);
        VectorXd p2 = xi_.block(idx2, 0, cdim_, 1);
        VectorXd diff = p1 - p2;
        double dist = diff.norm();
        double safety_dist = params_.robot_radius * 3.0;
        if (dist < safety_dist) {
          double violation = safety_dist - dist;
          interference_cost += violation * violation;
        }
      }
    }
  }
  total_cost += params_.mu * interference_cost;

  return total_cost;
}


void MultiChompNode::publish_state() {
  visualization_msgs::msg::MarkerArray marker_array;
  
  const int nq = params_.waypoints_per_robot;
  
  // a line strip marker for each robot's trajectory
  for (int r = 0; r < params_.num_robots; ++r) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "robot_" + std::to_string(r);
    marker.id = r;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    
    // cycle color for robots
    marker.color.a = 1.0;
    marker.color.r = (r % 3 == 0) ? 1.0 : 0.5;
    marker.color.g = (r % 3 == 1) ? 1.0 : 0.5;
    marker.color.b = (r % 3 == 2) ? 1.0 : 0.5;
    
    // points for each robot
    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      Eigen::Vector2d pt = xi_.block(idx, 0, cdim_, 1);
      
      geometry_msgs::msg::Point p;
      p.x = pt.x();
      p.y = pt.y();
      p.z = 0.0;
      marker.points.push_back(p);
    }
    
    marker_array.markers.push_back(marker);
  }
  
  marker_pub_->publish(marker_array);
}