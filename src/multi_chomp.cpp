#include "extended_spades/multi_chomp.hpp"

using std::placeholders::_1;

MultiChompNode::MultiChompNode() : Node("multi_chomp_server") {

    load_parameters();
    init_matrices();
    // ROS interface
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray> ("plan_markers", 10);
    rclcpp::QoS map_qos(1);
    map_qos.transient_local();

    // occupancy grid subscription
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>
        ("/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

    // timer (using action call for syncing the optimization step)
    // timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&MultiChompNode::solve_step, this));

    RCLCPP_INFO(this->get_logger(), "multi chomp server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters()
{
    this->declare_parameter<int>("num_robots", 2);
    this->declare_parameter<int>("waypoints_per_robot", 20);
    this->declare_parameter<double>("dt", 1.0);
    this->declare_parameter<double>("eta", 500.0);
    this->declare_parameter<double>("lambda", 10.0);
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
  
  RCLCPP_INFO(this->get_logger(), 
              "get_paths called: num_robots=%d, nq=%d, xidim_=%zu, xi_.size()=%ld",
              params_.num_robots, nq, xidim_, xi_.size());
  //guard against uninitialized optimizer state
  if (xidim_ == 0 || xi_.size() != static_cast<long>(xidim_)) {
      RCLCPP_ERROR(this->get_logger(), 
                   "Optimizer state invalid! xidim_=%zu, xi_.size()=%ld - returning empty paths",
                   xidim_, xi_.size());
      return out; 
  }

  for (int r = 0; r < params_.num_robots; ++r) {
    nav_msgs::msg::Path path;

    // use "map" frame and current time
    path.header.frame_id = "map"; 
    path.header.stamp = this->now();

    size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;
    path.poses.resize(nq);

    for (int k = 0; k < nq; ++k) {
      size_t idx = robot_offset + static_cast<size_t>(k) * cdim_;
      Eigen::Vector2d p = xi_.block(idx, 0, cdim_, 1);

      // orientation
      double yaw = 0.0;
      
      if (k < nq - 1) {
          size_t idx_next = robot_offset + static_cast<size_t>(k + 1) * cdim_;
          Eigen::Vector2d p_next = xi_.block(idx_next, 0, cdim_, 1);
          
          double dx = p_next.x() - p.x();
          double dy = p_next.y() - p.y();
          
          if (dx*dx + dy*dy > 0.0001) {
             yaw = std::atan2(dy, dx);
          } else if (k > 0) {
             size_t idx_prev = robot_offset + static_cast<size_t>(k - 1) * cdim_;
             Eigen::Vector2d p_prev = xi_.block(idx_prev, 0, cdim_, 1);
             yaw = std::atan2(p.y() - p_prev.y(), p.x() - p_prev.x());
          }
      } 
      else if (k > 0) {
          size_t idx_prev = robot_offset + static_cast<size_t>(k - 1) * cdim_;
          Eigen::Vector2d p_prev = xi_.block(idx_prev, 0, cdim_, 1);
          yaw = std::atan2(p.y() - p_prev.y(), p.x() - p_prev.x());
      }
      
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

    RCLCPP_INFO(this->get_logger(), 
                "Generated path for robot %d with %zu poses", r, path.poses.size());

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
    if (!map_received_) {
        return;
    }

    // --- Obstacle Gradients ---
    VectorXd nabla_obs = VectorXd::Zero(xidim_);
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        const int num_points = static_cast<int>(xidim_ / cdim_);
        
        for (int i = 0; i < num_points; ++i) {
            int idx = i * static_cast<int>(cdim_);
            VectorXd pt = xi_.block(idx, 0, cdim_, 1);
            
            Eigen::Vector2d grad_env;
            double dist = get_environment_distance(pt(0), pt(1), grad_env);

            if (dist < params_.obstacle_max_dist) {
                // Cost: c(x) = (d_max - d(x))^2
                // Gradient: dc/dx = -2*(d_max - d)*grad_d
                // grad_env points away from obstacles, so we ADD to push away
                double penetration = params_.obstacle_max_dist - dist;
                nabla_obs.block(idx, 0, cdim_, 1) += 2.0 * penetration * VectorXd(grad_env);
            }
        }
    }

    // --- Robot-Robot Interference Gradients ---
    VectorXd nabla_inter = VectorXd::Zero(xidim_);
    const double safety_dist = 2.0 * params_.robot_radius; // Consistent with cost
    
    for (int r1 = 0; r1 < params_.num_robots; ++r1) {
        for (int r2 = r1 + 1; r2 < params_.num_robots; ++r2) {
            for (int k = 0; k < params_.waypoints_per_robot; ++k) {
                int idx1 = (r1 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);
                int idx2 = (r2 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);

                VectorXd p1 = xi_.block(idx1, 0, cdim_, 1);
                VectorXd p2 = xi_.block(idx2, 0, cdim_, 1);
                VectorXd diff = p1 - p2;
                double dist = diff.norm();

                if (dist < safety_dist && dist > 1e-6) {
                    // Cost: c = (d_safe - ||p1-p2||)^2
                    // Gradient w.r.t p1: dc/dp1 = -2*(d_safe - dist)*(p1-p2)/dist
                    // Gradient w.r.t p2: dc/dp2 = +2*(d_safe - dist)*(p1-p2)/dist
                    double violation = safety_dist - dist;
                    VectorXd grad_force = 2.0 * violation * (diff / dist);
                    
                    nabla_inter.block(idx1, 0, cdim_, 1) -= grad_force;
                    nabla_inter.block(idx2, 0, cdim_, 1) += grad_force;
                }
            }
        }
    }

    // --- CHOMP Covariant Gradient Descent ---
    // Workspace gradient (obstacles + interference)
    VectorXd nabla_U = nabla_obs + params_.mu * nabla_inter;
    
    // Full gradient including smoothness
    VectorXd full_grad = params_.lambda * AAR_ * xi_ + nabla_U;
    
    // Covariant gradient step
    VectorXd dxi = AARinv_ * full_grad;
    
    // Update with step size
    xi_ -= (1.0 / params_.eta) * dxi;

    // --- Soft Boundary Constraints ---
    // Instead of hard reset, use strong attractive force toward start/goal
    const int nq = params_.waypoints_per_robot;
    const double boundary_stiffness = 1000.0; // High stiffness
    
    for (int r = 0; r < params_.num_robots; ++r) {
        size_t robot_offset = static_cast<size_t>(r) * nq * cdim_;

        // Start constraint
        if (r < static_cast<int>(start_states_.size())) {
            size_t start_idx = robot_offset;
            VectorXd error = xi_.block(start_idx, 0, cdim_, 1) - start_states_[r];
            xi_.block(start_idx, 0, cdim_, 1) -= boundary_stiffness * error / params_.eta;
        }

        // Goal constraint
        if (r < static_cast<int>(goal_states_.size())) {
            size_t goal_idx = robot_offset + static_cast<size_t>(nq - 1) * cdim_;
            VectorXd error = xi_.block(goal_idx, 0, cdim_, 1) - goal_states_[r];
            xi_.block(goal_idx, 0, cdim_, 1) -= boundary_stiffness * error / params_.eta;
        }
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
