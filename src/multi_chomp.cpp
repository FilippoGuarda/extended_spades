#include "extended_spades/multi_chomp.hpp"

using std::placeholders::_1;

MultiChompNode::MultiChompNode() : Node("multi_chomp_server") {
  load_parameters();
  init_matrices();

  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("plan_markers", 100);
  
  rclcpp::QoS map_qos(1);
  map_qos.transient_local();

  grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

  RCLCPP_INFO(this->get_logger(), "Multi-CHOMP server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters() {
  this->declare_parameter<int>("num_robots", 6);
  this->declare_parameter<int>("waypoints_per_robot", 100);
  this->declare_parameter<double>("dt", 0.1);
  this->declare_parameter<double>("eta", 1000.0); // update parameter is 1/eta
  this->declare_parameter<double>("lambda", 1.0);
  this->declare_parameter<double>("mu", 1.0);

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
}

void MultiChompNode::init_matrices() {
  size_t nq = params_.waypoints_per_robot; 
  AA_ = MatrixXd::Zero(nq * cdim_, nq * cdim_);
  
  for (size_t i=0; i < nq; ++i) { 
      AA_.block(cdim_ * i, cdim_ * i, cdim_, cdim_) = 2.0 * MatrixXd::Identity(cdim_, cdim_);
      if (i > 0) {
          AA_.block(cdim_ * (i - 1), cdim_ * i, cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_);
          AA_.block(cdim_ * i, cdim_ * (i - 1), cdim_, cdim_) = -1.0 * MatrixXd::Identity(cdim_, cdim_); 
      }
  }

  AAR_ = MatrixXd::Zero(xidim_, xidim_);
  for (int k = 0; k < params_.num_robots; ++k) {
      size_t offset = static_cast<size_t>(k) * nq * cdim_;
      AAR_.block(offset, offset, nq * cdim_, nq * cdim_) = AA_;
  }
  
  // Apply scaling and invert the full matrix
  // AAR_ /= (params_.dt * params_.dt * (nq + 1));
  AARinv_ = AAR_.inverse();
}

void MultiChompNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  current_map_ = *msg;
  update_distance_map(*msg);
  map_received_ = true;
}

void MultiChompNode::update_distance_map(const nav_msgs::msg::OccupancyGrid& grid) {
  map_resolution_ = grid.info.resolution;
  map_origin_x_ = grid.info.origin.position.x;
  map_origin_y_ = grid.info.origin.position.y;
  map_width_ = grid.info.width;
  map_height_ = grid.info.height;

  cv::Mat bin_img(map_height_, map_width_, CV_8UC1);

  for (int i = 0; i < map_height_; ++i) {
      int ros_row = (map_height_ - 1) - i;
      for (int j = 0; j < map_width_; ++j) {
          int8_t val = grid.data[ros_row * map_width_ + j];
          if (val == -1 || val > 50) {
              bin_img.at<uint8_t>(i, j) = 0; 
          } else {
              bin_img.at<uint8_t>(i, j) = 255;
          }
      }
  }

  cv::Mat dist_img_pixels;
  cv::distanceTransform(bin_img, dist_img_pixels, cv::DIST_L2, 5);
  dist_img_pixels.convertTo(dist_map_, CV_64F, map_resolution_);

  cv::Sobel(dist_map_, dist_grad_x_, CV_64F, 1, 0, 3);
  cv::Sobel(dist_map_, dist_grad_y_, CV_64F, 0, 1, 3);

  dist_grad_y_ = -dist_grad_y_;
}

double MultiChompNode::get_environment_distance(double x, double y, Eigen::Vector2d& gradient) const {
  if (dist_map_.empty()) {
      gradient << 0.0, 0.0;
      return 999.0;
  }

  const double min_x = map_origin_x_ + map_resolution_;
  const double max_x = map_origin_x_ + (map_width_ - 2) * map_resolution_;
  const double min_y = map_origin_y_ + map_resolution_;
  const double max_y = map_origin_y_ + (map_height_ - 2) * map_resolution_;

  if (x < min_x || x > max_x || y < min_y || y > max_y) {
      double grad_x = 0.0, grad_y = 0.0;
      double sq_pen_dist = 0.0;

      if (x < min_x) { grad_x = 1.0; sq_pen_dist += (min_x - x) * (min_x - x); } 
      else if (x > max_x) { grad_x = -1.0; sq_pen_dist += (x - max_x) * (x - max_x); }

      if (y < min_y) { grad_y = 1.0; sq_pen_dist += (min_y - y) * (min_y - y); } 
      else if (y > max_y) { grad_y = -1.0; sq_pen_dist += (y - max_y) * (y - max_y); }

      double norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
      gradient << grad_x / norm, grad_y / norm;
      return -std::sqrt(sq_pen_dist);
  }

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

std::vector<nav_msgs::msg::Path> MultiChompNode::get_paths(const std::vector<nav_msgs::msg::Path> & templates) const {
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
      } else {
          if (k > 0) {
              const auto& prev_q = path.poses[k-1].pose.orientation;
              yaw = std::atan2(2.0 * (prev_q.w * prev_q.z + prev_q.x * prev_q.y), 1.0 - 2.0 * (prev_q.y * prev_q.y + prev_q.z * prev_q.z));
          }
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

std::vector<Eigen::Vector2d> MultiChompNode::resample_path(const nav_msgs::msg::Path & path, int num_points) const {
  std::vector<Eigen::Vector2d> out;
  out.reserve(num_points);

  const size_t n = path.poses.size();
  if (n == 0 || num_points <= 0) return out;
  if (n == 1) {
      out.assign(num_points, Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
      return out;
  }

  std::vector<double> s(n, 0.0);
  for (size_t i = 1; i < n; ++i) {
      const auto & p0 = path.poses[i - 1].pose.position;
      const auto & p1 = path.poses[i].pose.position;
      s[i] = s[i - 1] + std::hypot(p1.x - p0.x, p1.y - p0.y);
  }
  
  double L = s.back();
  if (L < 1e-6) {
      out.assign(num_points, Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
      return out;
  }

  for (int k = 0; k < num_points; ++k) {
      double target_s = (k == num_points - 1) ? L : (static_cast<double>(k) / (num_points - 1)) * L;
      
      size_t idx = 0;
      while (idx < n - 1 && s[idx] < target_s) idx++;

      if (idx == 0) {
          out.push_back(Eigen::Vector2d(path.poses[0].pose.position.x, path.poses[0].pose.position.y));
          continue;
      }

      size_t prev_idx = idx - 1;
      double ds = s[idx] - s[prev_idx];
      double alpha = (ds > 1e-6) ? std::clamp((target_s - s[prev_idx]) / ds, 0.0, 1.0) : 0.0;

      const auto & p0 = path.poses[prev_idx].pose.position;
      const auto & p1 = path.poses[idx].pose.position;
      
      out.push_back(Eigen::Vector2d(p0.x + alpha * (p1.x - p0.x), p0.y + alpha * (p1.y - p0.y)));
  }
  return out;
}

bool MultiChompNode::set_paths(const std::vector<nav_msgs::msg::Path> & paths) {
  int new_num_robots = static_cast<int>(paths.size());
  const int nq = params_.waypoints_per_robot;

  if (nq <= 1) return false;

  if (new_num_robots != params_.num_robots) {
      params_.num_robots = new_num_robots;
      xidim_ = params_.num_robots * nq * cdim_;
      xi_ = VectorXd::Zero(xidim_);
      init_matrices(); 
  }

  start_states_.resize(params_.num_robots);
  goal_states_.resize(params_.num_robots);

  for (int r = 0; r < params_.num_robots; ++r) {
      const auto & path = paths[r];
      if (path.poses.size() < 2) return false;

      start_states_[r] = Eigen::Vector2d(path.poses.front().pose.position.x, path.poses.front().pose.position.y);
      goal_states_[r]  = Eigen::Vector2d(path.poses.back().pose.position.x,  path.poses.back().pose.position.y);

      auto samples = resample_path(path, nq);
      size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
      for (int k = 0; k < nq; ++k) {
          xi_.block(robot_offset + k * cdim_, 0, cdim_, 1) = samples[k];
      }
  }
  return true;
}

void MultiChompNode::solve_step() {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return;

  const int nq = params_.waypoints_per_robot;
  const double dt = params_.dt;

  // 1. Smoothness Gradient (Self-contained, NO bbR_)
  VectorXd nabla_smooth = AAR_ * xi_;

  // 2. Obstacle & Interference Gradients
  VectorXd nabla_obs = VectorXd::Zero(xidim_);
  VectorXd nabla_inter = VectorXd::Zero(xidim_);

  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      VectorXd start_p = start_states_[r];
      VectorXd end_p = goal_states_[r];

      for (int i = 0; i < nq; ++i) {
          int idx = offset + i * cdim_;
          VectorXd qq = xi_.block(idx, 0, cdim_, 1);

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
          if (vel < 1.0e-3) continue;

          VectorXd xdn = qd / vel;
          Eigen::Matrix2d prj = Eigen::Matrix2d::Identity() - xdn * xdn.transpose();

          VectorXd xdd = nabla_smooth.block(idx, 0, cdim_, 1);
          VectorXd kappa = (prj * xdd) / (vel * vel);

          Eigen::Vector2d grad_env;
          double dist = get_environment_distance(qq(0), qq(1), grad_env);

          if (dist < params_.obstacle_max_dist) {
              double gain = 2.0; 
              double term = (1.0 - dist / params_.obstacle_max_dist);
              double cost = gain * params_.obstacle_max_dist * std::pow(term, 3.0) / 3.0;

              VectorXd delta = -gain * std::pow(term, 2.0) * VectorXd(grad_env);
              nabla_obs.block(idx, 0, cdim_, 1) += vel * (prj * delta - cost * kappa);
          }

          const double safety_dist = 2.0 * params_.robot_radius;
          const double gainR = 1.0;

          for (int r2 = 0; r2 < params_.num_robots; ++r2) {
              if (r == r2) continue;
              size_t offset2 = static_cast<size_t>(r2) * nq * cdim_;
              
              VectorXd q2 = xi_.block(offset2 + i * cdim_, 0, cdim_, 1);
              VectorXd diff = qq - q2;
              double ddnorm = diff.norm();

              if (ddnorm < safety_dist && ddnorm > 1e-9) {
                   double termR = (1.0 - ddnorm / safety_dist);
                   double costR = gainR * safety_dist * std::pow(termR, 3.0) / 3.0;

                   VectorXd deltaR = -gainR * std::pow(termR, 2.0) * (diff / ddnorm);
                   nabla_inter.block(idx, 0, cdim_, 1) += vel * (prj * deltaR - costR * kappa);
              }
          }
      }
  }

  // 3. Combine 
  VectorXd full_grad = nabla_obs + params_.lambda * nabla_smooth + params_.mu * nabla_inter;

  // Clip gradient before preconditioning to prevent wild values
  double g_norm = full_grad.norm();
  if (g_norm > 1e6) full_grad *= (1e6 / g_norm);

  // 4. Precondition
  VectorXd dxi = AARinv_ * full_grad;

  // CRITICAL FIX: The preconditioner AARinv_ spans the entire trajectory. 
  // Because the boundary rows of AAR_ couple with the interior, AARinv_ WILL inject 
  // non-zero update values into the boundary elements of dxi. 
  // We MUST explicitly zero out the boundary components of the step BEFORE applying it.
  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      dxi.block(offset, 0, cdim_, 1).setZero();
      dxi.block(offset + (nq - 1) * cdim_, 0, cdim_, 1).setZero();
  }

  // Apply update
  xi_ -= dxi / params_.eta;

  // 5. Hard Constraints enforcement (Safety net)
  for (int r = 0; r < params_.num_robots; ++r) {
      size_t offset = static_cast<size_t>(r) * nq * cdim_;
      xi_.block(offset, 0, cdim_, 1) = start_states_[r];
      xi_.block(offset + (nq-1)*cdim_, 0, cdim_, 1) = goal_states_[r];
  }

  publish_state();
}

double MultiChompNode::compute_current_cost() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  if (!map_received_) return 0.0;

  double total_cost = 0.0;

  VectorXd smooth_term = AAR_ * xi_;
  double smoothness_cost = 0.5 * xi_.dot(smooth_term);
  total_cost += params_.lambda * smoothness_cost;

  const int nq = params_.waypoints_per_robot;
  double obstacle_cost = 0.0;
  double interference_cost = 0.0;
  const double safety_dist = 2.0 * params_.robot_radius;

  for (int r1 = 0; r1 < params_.num_robots; ++r1) {
    for (int k = 1; k < nq - 1; ++k) {
      int idx1 = (r1 * nq + k) * static_cast<int>(cdim_);
      VectorXd p1 = xi_.block(idx1, 0, cdim_, 1);
      
      Eigen::Vector2d grad_env;
      double dist = get_environment_distance(p1(0), p1(1), grad_env);
      if (dist < params_.obstacle_max_dist) {
        double penetration = params_.obstacle_max_dist - dist;
        obstacle_cost += penetration * penetration;
      }

      for (int r2 = r1 + 1; r2 < params_.num_robots; ++r2) {
        int idx2 = (r2 * nq + k) * static_cast<int>(cdim_);
        VectorXd p2 = xi_.block(idx2, 0, cdim_, 1);
        double r_dist = (p1 - p2).norm();
        if (r_dist < safety_dist) {
          double violation = safety_dist - r_dist;
          interference_cost += violation * violation;
        }
      }
    }
  }

  total_cost += obstacle_cost + params_.mu * interference_cost;
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
