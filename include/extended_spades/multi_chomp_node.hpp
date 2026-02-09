#ifndef EXTENDED_SPADES__MULTI_CHOMP_NODE_HPP_
#define EXTENDED_SPADES__MULTI_CHOMP_NODE_HPP_

#include <vector>
#include <mutex>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msg/msg/marker_array.hpp"
#include "geometry_msg/point.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

struct ChompParameters {
    double dt = 1.0;
    double eta = 100.0;
    double lambda = 1.0;
    double mu = 0.4;
    double robot_radius = 0.5;
    double obstacle_max_dist = 50.0;
    int num_robots = 2;
    int waypoints_per_robot = 20;
};

class MultiChompNode : public rclcpp::Node {

public MultiChompNode();

    void solve_step();

private:
    // visualization publisher
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // loop timer
    rclcpp::TimerBase::SharedPtr timer_;

    // costmap subscriber
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
    // state protection

    std::mutex map_mutex_;
    nav_msgs::msg::OccupancyGrid current_map_;
    bool map_received_ = false;

    // state
    ChompParameters params_;

    // distance field state
    cv::Mat dist_map_;
    cv::Mat dist_grad_x_;
    cv::Mat dist_grad_y_;
    float map_resolution_;
    double map_origin_x_, map_origin_y_;
    int map_width_; map_height_;
    void update_distance_map(const nav_msgs::msg::OccupancyGrid& grid);
    

    // trajectory state
    size_t cdim_ = 2; // 2D (number of dimensions)
    size_t xidim_; 
    Vector xi_;  // trajectory vector

    Matrix AA_; // metrix matrix (cost)
    Matrix AAR_; // multiple robots AA
    Matrix AARinv_; // AAR inverse 
    Vector bbR_; // acceleration bias

    // state scenarios 
    std::vector::<Vector> start_states_;
    std::vector::<Vector> goal_states_;

    void map_callback(const nav_msg::msg::OccupancyGrid::SharedPtr msg);

    double get_environment_distance(double x, double y, Eigen::Vector2d& gradient);

    // helpers initialization
    void load_parameters();
    void init_matrices();

    void publish_state();

}


#endif EXTENDED_SPADES__MULTI_CHOMP_NODE_HPP_