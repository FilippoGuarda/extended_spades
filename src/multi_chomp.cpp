#include "multi_chomp_node.hpp"

using std::placeholders::_1;

MultiChompNode::MultiChompNode() : Node("extended_spades_server") {

    load_parameters();
    init_matrices();
    // ROS interface
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray> ("plan_markers", 10);
    rclcpp::QoS map_qos(1);
    map_qos.transient_local();

    // occupancy grid subscription
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>
        ("/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

    // timer
    timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&MultiChompNode::solve_step, this));

    RCLCPP_INFO(this->get_logger(), "Extended SPADES server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters()
{
    this->declare_parameter<int>("num_robots", 2);
    this->declare_parameter<int>("waypoints_per_robot", 20);
    this->declare_parameter<double>("dt", 1.0);
    this->declare_parameter<double>("eta", 100.0);
    this->declare_parameter<double>("lambda", 1.0);
    this->declare_parameter<double>("mu", 0.4);

    params_.num_robots = this->get_parameter("num_robots").as_int();
    params_.waypoints_per_robot = this->get_parameter("waypoints_per_robot").as_int();
    params_.dt = this->get_parameter("dt").as_double();
    params_.eta = this->get_parameter("eta").as_double();
    params_.lambda = this->get_parameter("lambda").as_double();
    params_.mu = this->get_parameter("mu").as_double();

    xidim_ = params_.num_robots * params_.waypoints_per_robot * cdim_;
    xi_  = Vector::Zero(xidim_);
    bbR_ = Vector::Zero(xidim_);
}

void MultiChompNode::init_matrices() {

    // single robot metric A from waypoints
    size_t nq = params_.waypoints_per_robot; 
    AA_ = Matrix::Zero(nq*cdim_, nq*cdim_);
    
    for (size_t i=0; i < nq; ++i) { 
        AA_.block(cdim_ * i, cdim_ * i, cdim_, cdim_) = 2.0 * Matrix::Identity(cdim_, cdim_);

        // outside of diagonal
        if (i > 0) {
            AA_.block(cdim_ * (i - 1), cdim_ * i, cdim_, cdim_) = - 1.0 * Matrix::Identity(cdim_, cdim_);
            AA_.block(cdim_ * i, cdim_ * (i - 1), cdim_, cdim_) = - 1.0 * Matrix::Identity(cdim_, cdim_); 
        }
    }

    // now create multi robot metric, each block is one robot smoothness cost
    // robots only intaract through the interference objective (computed in solver)
    AAR_ = Matrix::Zero(xidim_, xidim_);
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

double MultiChompNode::get_environment_distance(double x, double y, Eigen::Vector2d& gradient)
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

void MultiChompNode::solve_step() {
    if (!map_received_) {
        return;
    }

    Vector nabla_smooth = AAR_ * xi_ + bbR_;
    // obstacle and interference gradients
    Vector nabla_obs = Vector::Zero(xidim_);
    Vector nabla_inter = Vector::Zero(xidim_);
    // robot pairs iteration
    for (int r1 = 0; r1 < params_.num_robots; ++r1) {
        for (int r2 = r1 + 1; r2 < params_.num_robots; ++r2) {
            for (int k = 0; k < params_.waypoints_per_robot; ++k) {
                int idx1 = (r1 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);
                int idx2 = (r2 * params_.waypoints_per_robot + k) * static_cast<int>(cdim_);

                Vector p1 = xi_.block(idx1, 0, cdim_, 1);
                Vector p2 = xi_.block(idx2, 0, cdim_, 1);

                Vector diff = p1 - p2;
                double dist = diff.norm();
                double safety_dist = params_.robot_radius * 3.0;

                if (dist < safety_dist && dist > 1e-6) {
                    // update safety distance gradient
                    Vector grad_force = -1.0 * (safety_dist - dist) * (diff / dist);

                    nabla_inter.block(idx1, 0, cdim_, 1) += grad_force;
                    nabla_inter.block(idx2, 0, cdim_, 1) -= grad_force;
                }
            }
        }
    }

    // create mutex lock to reduce lock scope
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        // one configuration per waypoint
        const int num_points = static_cast<int>(xidim_ / cdim_);
        for (int i = 0; i < num_points; ++i){
            int idx = i * static_cast<int>(cdim_);
            Vector pt = xi_.block(idx, 0, cdim_, 1);
            
            Eigen::Vector2d grad_env;
            double dist = get_environment_distance(pt(0), pt(1), grad_env);

            if (dist < params_.obstacle_max_dist) {
                double weight = (params_.obstacle_max_dist - dist);
                nabla_obs.block(idx, 0, cdim_, 1) -= weight * Vector(grad_env);
            }
        }
    }

    Vector total_grad = nabla_obs + params_.lambda * nabla_smooth + params_.mu * nabla_inter;
    Vector dxi = AARinv_ * total_grad;
    xi_ -= (1.0 / params_.eta) * dxi;

    publish_state();

}
