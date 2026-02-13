#ifndef EXTENDED_SPADES__EXTENDED_SPADES_ACTION_SERVER_HPP_
#define EXTENDED_SPADES__EXTENDED_SPADES_ACTION_SERVER_HPP_

#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "extended_spades/action/multi_chomp_optimize.hpp"
#include "nav_msgs/msg/path.hpp"

#include "multi_chomp.hpp"

class ExtendedSpadesActionServer : public rclcpp::Node
{
public:
    using MultiChompOptimize = extended_spades::action::MultiChompOptimize;
    using GoalHandleMultiChomp = rclcpp_action::ServerGoalHandle<MultiChompOptimize>;

    explicit ExtendedSpadesActionServer(const rclcpp::NodeOptions & options =
                                        rclcpp::NodeOptions());

private: 
    rclcpp_action::Server<MultiChompOptimize>::SharedPtr action_server_;

    std::shared_ptr<MultiChompNode> optimizer_; 

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const MultiChompOptimize::Goal> goal);

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleMultiChomp> goal_handle);

    void handle_accepted(
        const std::shared_ptr<GoalHandleMultiChomp> goal_handle);
    
    void execute_goal(
        const std::shared_ptr<GoalHandleMultiChomp> goal_handle);

    bool load_paths_into_state( 
        const std::vector<nav_msgs::msg::Path> & paths
    );

    std::vector<nav_msgs::msg::Path> export_state_to_paths(
        const std::vector<nav_msgs::msg::Path> & template_paths) const;
    
};

#endif //EXTENDED_SPADES__EXTENDED_SPADES_ACTION_SERVER_HPP_