#include "extended_spades/multi_chomp.hpp"

using std::placeholders::_1;

MultiChompNode::MultiChompNode() : Node("multi_chomp_server") {
  load_parameters();
  init_matrices();

  // ROS interface
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("plan_markers", 20);

  rclcpp::QoS map_qos(1);
  map_qos.transient_local();

  // occupancy grid subscription
  grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    "/robot1/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

  RCLCPP_INFO(this->get_logger(), "Multi-CHOMP server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters()
{
  this->declare_parameter<int>("num_robots", 6);
  this->declare_parameter<int>("waypoints_per_robot", 20);
  this->declare_parameter<double>("dt", 1.0);
  this->declare_parameter<double>("eta", 0.5);
  this->declare_parameter<double>("alpha", 1.0); 
  this->declare_parameter<double>("lambda", 1.0);
  this->declare_parameter<double>("mu", 1.0);

  params_.robot_radius = 0.5;
  params_.obstacle_max_dist = 1.0;
  params_.num_robots = this->get_parameter("num_robots").as_int();
  params_.waypoints_per_robot = this->get_parameter("waypoints_per_robot").as_int();
  params_.dt = this->get_parameter("dt").as_double();
  params_.eta = this->get_parameter("eta").as_double();
  params_.alpha = this->get_parameter("alpha").as_double();
  params_.lambda = this->get_parameter("lambda").as_double();
  params_.mu = this->get_parameter("mu").as_double();

  xidim_ = params_.num_robots * params_.waypoints_per_robot * cdim_;
  xi_ = VectorXd::Zero(xidim_);
  xi_init_ = VectorXd::Zero(xidim_);
  bbR_ = VectorXd::Zero(xidim_);
}

void MultiChompNode::init_matrices() {
  const size_t nq = params_.waypoints_per_robot;
  const size_t full_dim = nq * cdim_;
  const size_t interior_dim = (nq - 2) * cdim_;

  // Full finite difference matrix (all nq waypoints)
  AA_ = MatrixXd::Zero(full_dim, full_dim);
  for (size_t i = 0; i < nq; ++i) {
    AA_.block(cdim_ * i, cdim_ * i, cdim_, cdim_) = 2.0 * MatrixXd::Identity(cdim_, cdim_);
    if (i > 0) {
      AA_.block(cdim_ * (i-1), cdim_ * i, cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_);
      AA_.block(cdim_ * i, cdim_ * (i-1), cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_);
    }
  }

  AAR_ = MatrixXd::Zero(xidim_, xidim_);
  for (int k = 0; k < params_.num_robots; ++k) {
    size_t offset = static_cast<size_t>(k) * nq * cdim_;
    AAR_.block(offset, offset, full_dim, full_dim) = AA_;
  }

  // Correct preconditioner: invert the FULL AA_ first, then extract the interior block
  // This ensures AAinv_interior_ * (AA_interior * v) ≈ v for interior vectors
  MatrixXd AA_inv_full = AA_.inverse();
  AAinv_interior_ = AA_inv_full.block(cdim_, cdim_, interior_dim, interior_dim);
}


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

  bool treat_unknown_as_free = false; 

  for (int i = 0; i < map_height_; ++i) {
    int ros_row = (map_height_ - 1) - i;  // flip vertical axis
    for (int j = 0; j < map_width_; ++j) {
      int8_t val = grid.data[ros_row * map_width_ + j];
      
      if (val == -1) {
        // Handle Unknown Space
        bin_img.at<uint8_t>(i, j) = treat_unknown_as_free ? 255 : 0;
      } else if (val > 50) {
        // Handle Occupied Space (Obstacle)
        // 0 is the "zero-distance" target for distanceTransform
        bin_img.at<uint8_t>(i, j) = 0;
      } else {
        // Handle Free Space
        bin_img.at<uint8_t>(i, j) = 255;
      }
    }
  }

  cv::Mat dist_img_pixels;
  // Compute distance to the nearest 0 pixel
  cv::distanceTransform(bin_img, dist_img_pixels, cv::DIST_L2, 5);

  dist_img_pixels.convertTo(dist_map_, CV_64F, map_resolution_);

  cv::Sobel(dist_map_, dist_grad_x_, CV_64F, 1, 0, 3);
  cv::Sobel(dist_map_, dist_grad_y_, CV_64F, 0, 1, 3);

  // Sobel dy is in image coordinates (row increases downward)
  // but world y increases upward — flip the y gradient sign
  dist_grad_y_ = -dist_grad_y_;
}


double MultiChompNode::get_environment_distance(double x, double y, Eigen::Vector2d& gradient) const
{
  if (dist_map_.empty()) {
    gradient << 0.0, 0.0;
    return 999.0; // Return free space if map is missing to prevent arbitrary explosions
  }

  // Define the safe 1-pixel interior boundary in world coordinates
  const double min_x = map_origin_x_ + map_resolution_;
  const double max_x = map_origin_x_ + (map_width_ - 2) * map_resolution_;
  const double min_y = map_origin_y_ + map_resolution_;
  const double max_y = map_origin_y_ + (map_height_ - 2) * map_resolution_;

  // If outside the safe map bounds, synthesize a steep restoring gradient
  if (x < min_x || x > max_x || y < min_y || y > max_y) {
    double grad_x = 0.0;
    double grad_y = 0.0;
    double sq_pen_dist = 0.0;

    if (x < min_x) {
      grad_x = 1.0; // Point right (towards map)
      sq_pen_dist += (min_x - x) * (min_x - x);
    } else if (x > max_x) {
      grad_x = -1.0; // Point left (towards map)
      sq_pen_dist += (x - max_x) * (x - max_x);
    }

    if (y < min_y) {
      grad_y = 1.0; // Point up (towards map)
      sq_pen_dist += (min_y - y) * (min_y - y);
    } else if (y > max_y) {
      grad_y = -1.0; // Point down (towards map)
      sq_pen_dist += (y - max_y) * (y - max_y);
    }

    double norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
    gradient << grad_x / norm, grad_y / norm;

    // Return negative distance to trigger massive cubic cost scaling
    return -std::sqrt(sq_pen_dist);
  }

  // Normal inside-map calculation
  double gx = (x - map_origin_x_) / map_resolution_;
  double gy = (double)(map_height_ - 1) - (y - map_origin_y_) / map_resolution_;

  int u = static_cast<int>(gx);
  int v = static_cast<int>(gy);

  double dist = dist_map_.at<double>(v, u);
  double dx   = dist_grad_x_.at<double>(v, u);
  double dy   = dist_grad_y_.at<double>(v, u);

  double norm = std::sqrt(dx * dx + dy * dy);
  if (norm > 1e-5) {
    gradient << dx / norm, dy / norm;
  } else {
    gradient << 0.0, 0.0;
  }

  return dist;
}

std::vector<nav_msgs::msg::Path>
MultiChompNode::get_paths(const std::vector<nav_msgs::msg::Path> & templates) const
{
  std::vector<nav_msgs::msg::Path> out;
  out.resize(params_.num_robots);

  const int nq = params_.waypoints_per_robot;

  if (xidim_ == 0 || xi_.size() != static_cast<long>(xidim_)) {
    return out;
  }

  for (int r = 0; r < params_.num_robots; ++r) {
    nav_msgs::msg::Path path;
    path.header.frame_id = "map";
    path.header.stamp = this->now();

    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    path.poses.resize(nq);

    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      Eigen::Vector2d p = xi_.block(idx, 0, cdim_, 1);

      double yaw = 0.0;
      Eigen::Vector2d p_next = p;
      Eigen::Vector2d p_prev = p;
      bool found_next = false;
      bool found_prev = false;

      for(int step = 1; step <= 3 && (k + step) < nq; ++step) {
        size_t idx_n = robot_offset + static_cast<size_t>(k + step) * cdim_;
        Eigen::Vector2d check = xi_.block(idx_n, 0, cdim_, 1);
        if ((check - p).squaredNorm() > 0.01) {
          p_next = check;
          found_next = true;
          break;
        }
      }

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
        yaw = std::atan2(p_next.y() - p_prev.y(), p_next.x() - p_prev.x());
      } else if (found_next) {
        yaw = std::atan2(p_next.y() - p.y(), p_next.x() - p.x());
      } else if (found_prev) {
        yaw = std::atan2(p.y() - p_prev.y(), p.x() - p_prev.x());
      }

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

std::vector<Eigen::Vector2d>
MultiChompNode::resample_path(const nav_msgs::msg::Path & path, int num_points) const
{
  std::vector<Eigen::Vector2d> out;
  out.reserve(num_points);

  const size_t n = path.poses.size();
  if (n == 0 || num_points <= 0) {
    return out;
  }
  if (n == 1) {
    Eigen::Vector2d p(path.poses[0].pose.position.x,
                      path.poses[0].pose.position.y);
    out.assign(num_points, p);
    return out;
  }

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
    Eigen::Vector2d p(path.poses[0].pose.position.x, path.poses[0].pose.position.y);
    out.assign(num_points, p);
    return out;
  }

  for (int k = 0; k < num_points; ++k) {
    // Exact clamp to ensure the last point is exactly L
    double target_s = (k == num_points - 1) ? L : 
                      (static_cast<double>(k) / static_cast<double>(num_points - 1)) * L;
    
    // Find the first index where the cumulative distance is greater than or equal to target_s
    size_t idx = 0;
    while (idx < n - 1 && s[idx] < target_s) {
        idx++;
    }

    // If we hit the exact first point or somehow underflowed
    if (idx == 0) {
        out.push_back(Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
        continue;
    }

    size_t prev_idx = idx - 1;
    double ds = s[idx] - s[prev_idx];
    
    // Robustly calculate alpha, guarding against identical poses
    double alpha = 0.0;
    if (ds > 1e-6) {
        alpha = (target_s - s[prev_idx]) / ds;
        // Clamp alpha to [0, 1] to prevent floating point extrapolation
        alpha = std::max(0.0, std::min(1.0, alpha)); 
    }

    const auto & p0 = path.poses[prev_idx].pose.position;
    const auto & p1 = path.poses[idx].pose.position;
    
    Eigen::Vector2d p;
    p.x() = p0.x + alpha * (p1.x - p0.x);
    p.y() = p0.y + alpha * (p1.y - p0.y);
    out.push_back(p);
  }
  
  return out;
}


bool MultiChompNode::set_paths(const std::vector<nav_msgs::msg::Path>& paths)
{
  int new_num_robots = static_cast<int>(paths.size());
  const int nq = params_.waypoints_per_robot;

  if (nq <= 1) {
    RCLCPP_ERROR(this->get_logger(), "set_paths: waypoints_per_robot must be > 1");
    return false;
  }

  if (new_num_robots != params_.num_robots) {
    RCLCPP_INFO(this->get_logger(), "Resizing for %d robots", new_num_robots);
    params_.num_robots = new_num_robots;
    xidim_ = static_cast<size_t>(params_.num_robots) * nq * cdim_;

    xi_.conservativeResizeLike(VectorXd::Zero(xidim_));
    xi_init_.conservativeResizeLike(VectorXd::Zero(xidim_));
    bbR_.conservativeResizeLike(VectorXd::Zero(xidim_));

    size_t total_nq = static_cast<size_t>(params_.num_robots) * nq;
    AAR_ = MatrixXd::Zero(xidim_, xidim_);
    for (int k = 0; k < params_.num_robots; ++k) {
      size_t offset = static_cast<size_t>(k) * nq * cdim_;
      AAR_.block(offset, offset, nq * cdim_, nq * cdim_) = AA_;
    }
  }

  start_states_.resize(params_.num_robots);
  goal_states_.resize(params_.num_robots);

  for (int r = 0; r < params_.num_robots; ++r) {
    const auto& path = paths[r];
    if (path.poses.size() < 2) {
      RCLCPP_ERROR(this->get_logger(), "set_paths: robot %d path has < 2 poses", r);
      return false;
    }

    const auto& p_start = path.poses.front().pose.position;
    const auto& p_goal  = path.poses.back().pose.position;
    start_states_[r] = Eigen::Vector2d(p_start.x, p_start.y);
    goal_states_[r]  = Eigen::Vector2d(p_goal.x,  p_goal.y);

    auto samples = resample_path(path, nq);
    if (static_cast<int>(samples.size()) != nq) {
      RCLCPP_ERROR(this->get_logger(), "set_paths: resample failed for robot %d", r);
      return false;
    }

    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      xi_.block(idx, 0, cdim_, 1)      = samples[k];
      xi_init_.block(idx, 0, cdim_, 1) = samples[k];
    }
  }

  RCLCPP_INFO(this->get_logger(), "Loaded %d robot paths", params_.num_robots);
  return true;
}

void MultiChompNode::solve_step() {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return;

  const int nq = params_.waypoints_per_robot;
  const double dt = params_.dt;

  // 1. Fidelity gradient
  VectorXd nabla_fidelity = params_.alpha * (xi_ - xi_init_);
  const size_t total_nq = static_cast<size_t>(params_.num_robots) * nq;
  const double scaling_factor = 1.0 / (dt * dt * (total_nq + 1));

  bbR_ = VectorXd::Zero(xidim_);
  for (int r = 0; r < params_.num_robots; ++r) {
    size_t ro = static_cast<size_t>(r) * nq * cdim_;
    bbR_.block(ro + 1 * cdim_, 0, cdim_, 1) =
        -scaling_factor * start_states_[r];
    bbR_.block(ro + static_cast<size_t>(nq - 2) * cdim_, 0, cdim_, 1) =
        -scaling_factor * goal_states_[r];
  }

  // 2. Smoothness gradient - bbR_ removed as AAR_*xi_ intrinsically handles boundaries
  // TODO: check if bbR_ is necessary for correct update
  VectorXd nabla_smooth = -params_.lambda * (AAR_ * xi_);

  // 3. Obstacle & Interference gradients
  VectorXd nabla_obs   = VectorXd::Zero(xidim_);
  VectorXd nabla_inter = VectorXd::Zero(xidim_);

  for (int r = 0; r < params_.num_robots; ++r) {
    size_t offset = static_cast<size_t>(r) * nq * cdim_;

    for (int i = 1; i < nq - 1; ++i) {
      int idx = offset + i * cdim_;
      VectorXd qq = xi_.block(idx, 0, cdim_, 1);

      VectorXd q_next = xi_.block(idx + cdim_, 0, cdim_, 1);
      VectorXd q_prev = xi_.block(idx - cdim_, 0, cdim_, 1);
      VectorXd qd = (q_next - q_prev) / (2.0 * dt);

      double vel = qd.norm();
      if (vel < 1.0e-3) continue;

      VectorXd xdn = qd / vel;
      Eigen::Matrix2d prj = Eigen::Matrix2d::Identity() - xdn * xdn.transpose();

      VectorXd q_acc = (q_next - 2.0 * qq + q_prev) / (dt * dt);
      VectorXd kappa = (prj * q_acc) / (vel * vel);

      // --- Obstacle gradient (Eq. 5) ---
      Eigen::Vector2d grad_env;
      double dist = get_environment_distance(qq(0), qq(1), grad_env);

      if (dist < params_.obstacle_max_dist) {
        double gain = 2.0;
        double term = (1.0 - dist / params_.obstacle_max_dist);
        double cost = gain * params_.obstacle_max_dist * pow(term, 3.0) / 3.0;
        VectorXd delta = -gain * pow(term, 2.0) * VectorXd(grad_env);
        nabla_obs.block(idx, 0, cdim_, 1) += vel * (prj * delta - cost * kappa);
      }

      // --- Interference gradient (Eq. 9) ---
      double t = static_cast<double>(i) / static_cast<double>(nq - 1);
      double boundary_weight = std::min(t, 1.0 - t) * 2.0;

      const double safety_dist = 2.0 * params_.robot_radius;

      for (int r2 = 0; r2 < params_.num_robots; ++r2) {
        if (r == r2) continue;
        size_t offset2 = static_cast<size_t>(r2) * nq * cdim_;
        VectorXd q2 = xi_.block(offset2 + i * cdim_, 0, cdim_, 1);
        VectorXd diff = qq - q2;
        double dd_norm = diff.norm();

        if (dd_norm < safety_dist && dd_norm > 1e-9) {
          double term_R   = 1.0 - dd_norm / safety_dist;
          double cost_R   = safety_dist * pow(term_R, 3.0) / 3.0;
          VectorXd delta_R = -pow(term_R, 2.0) * (diff / dd_norm);
          nabla_inter.block(idx, 0, cdim_, 1) +=
              boundary_weight * vel * (prj * delta_R - cost_R * kappa);
        }
      }
    }
  }

  // 4. Combined gradient
  VectorXd full_grad =  nabla_smooth + nabla_obs +
                       params_.mu * nabla_inter;

  for (int r = 0; r < params_.num_robots; ++r) {
    size_t offset = static_cast<size_t>(r) * nq * cdim_;
    full_grad.block(offset, 0, cdim_, 1).setZero();
    full_grad.block(offset + static_cast<size_t>(nq - 1) * cdim_, 0, cdim_, 1).setZero();
  }

  double g_norm = full_grad.norm();
  if (g_norm > 1e6) full_grad *= (1e6 / g_norm);

  // 5. Per-robot interior-only preconditioning
  VectorXd dxi = VectorXd::Zero(xidim_);
  size_t interior_dim = static_cast<size_t>(nq - 2) * cdim_;
  for (int r = 0; r < params_.num_robots; ++r) {
    size_t offset = static_cast<size_t>(r) * nq * cdim_;
    VectorXd grad_interior = full_grad.block(offset + cdim_, 0, interior_dim, 1);
    dxi.block(offset + cdim_, 0, interior_dim, 1) = AAinv_interior_ * grad_interior;
  }

  xi_ -= dxi * params_.eta;

  // 6. Hard constraints — enforce start/goal every iteration
  for (int r = 0; r < params_.num_robots; ++r) {
    size_t offset = static_cast<size_t>(r) * nq * cdim_;
    xi_.block(offset, 0, cdim_, 1) = start_states_[r];
    xi_.block(offset + static_cast<size_t>(nq - 1) * cdim_, 0, cdim_, 1) = goal_states_[r];
  }

  // 8. Map boundary clamp on interior waypoints
  if (!dist_map_.empty()) {
    const double margin = params_.robot_radius;
    const double min_x = map_origin_x_ + margin;
    const double max_x = map_origin_x_ + map_width_  * map_resolution_ - margin;
    const double min_y = map_origin_y_ + margin;
    const double max_y = map_origin_y_ + map_height_ * map_resolution_ - margin;

    for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      for (int k = 1; k < nq - 1; ++k) {
        double & x = xi_(offset + k * cdim_);
        double & y = xi_(offset + k * cdim_ + 1);
        x = std::max(min_x, std::min(max_x, x));
        y = std::max(min_y, std::min(max_y, y));
      }
    }
  }

  publish_state();
}


double MultiChompNode::compute_current_cost() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return 0.0;

  double total_cost = 0.0;

  VectorXd diff = xi_ - xi_init_;
  double fidelity_cost = 0.5 * params_.alpha * diff.dot(diff);
  total_cost += fidelity_cost;

  // Smoothness Cost (scaled by λ) TODO: check if this is not the cause of segfault
  VectorXd smooth_term = AAR_ * xi_ + bbR_;
  double smoothness_cost = 0.5 * params_.lambda * xi_.dot(smooth_term);
  total_cost += smoothness_cost;

  // Obstacle Cost
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

  // Interference Cost
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

  for (int r = 0; r < params_.num_robots; ++r) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "robot_" + std::to_string(r);
    marker.id = r;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;

    marker.color.a = 1.0;
    marker.color.r = (r % 3 == 0) ? 1.0 : 0.5;
    marker.color.g = (r % 3 == 1) ? 1.0 : 0.5;
    marker.color.b = (r % 3 == 2) ? 1.0 : 0.5;

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
