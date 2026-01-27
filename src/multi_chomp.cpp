#include "multi_chomp_node.hpp"

using std::placeholders::_1;

MultiChompNode:MultiChompNode() : Node("extended_spades_server") {

    load_parameters();
    init_matrices();
    init_default_scenario();
    // ROS interface
    marker_pub = this->create_publisher<visualization_msgs::msg::MarkerArray> ("plan_markers", 10);
    rclcpp::QoS map_qos(1);
    map_qos.transient_local();

    // occupancy grid subscription
    grid_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>
        ("/global_costmap/costmap", map_qos, std::bind(&MultiChompNode::map_callback, this, _1));

    // timer
    timer_ = this->create_wall_timer(std::chrono::milliseconds(33), std::bind(&MultiChompNode::solve_step, this));

    RCLCPP_INFO(this->get_logger(), "Extended SPADES server initialized for %d robots", params_.num_robots);
}

void MultiChompNode::load_parameters() {

    // params default values
    this->declare_parameters("num_robots", 2);
    this->declare_parameters("waypoints_per_robot", 20);
    this->declare_parameters("dt", 1.0);
    this->declare_parameters("eta", 100.0);
    this->declare_parameters("lambda", 1.0);
    this->declare_parameters("mu", 0.4);

    // get values from launch file
    params_.num_robots = this->get_parameter("num_robots").as_int();
    params_.waypoint_per_robot = this->get_parameter("waypoints_per_robot").as_int();
    params_.dt = this->get_parameter("dt").as_double();
    params_.eta = this->get_parameter("eta").as_double();
    params_.lambda = this->get_parameter("lambda").as_double(); 
    params_.mu = this->get_parameter("mu").as_double();

    // xidim must be computed from params
    xidim_ = params.num_robots * params_.waypoints_per_robot * cdim_;
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
    for (int k; k < params_.num_robots; ++k){ 
        size_t offset = k * (nq * cdim_);
        AAR_.block(offset, offset, nq*cdim_, nq*cdim_);
    }

    // scale with time step
    AAR_ /= (params_.dt * params_.dt * (nq + 1));

    // IMPORTANT!!!!!! <-----------------------------------------------------------------------
    // pre compute inverse for update xi = xi - (1/eta) * Ainv * Gradient
    // if N > 50 you might want to invert the block diagonal matrix instead of the full thing
    AARinv_ = AAR_.inverse();
}

// void MultiChompNode::init_default_scenario() {
    // 
// }

// map callback placeholder
void MultiChompNode::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {

    std::lock_guard<std::mutex> lock(map_mutex_);
    current_map_ = *msg;
    map_received_ = true;
}

