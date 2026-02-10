#include "extended_spades_action_server.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

ExtendedSpadesActionServer::ExtendedSpadesActionServer(
    const rclcpp::NodeOptions & options) 
    : Node("extended_spades_action_server", options)
{
    // create optimizer component
    optimizer_ = std::make_shared<MultiChompNode>();

    auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    exec->add_node(optimizer_);
    std::thread([exec]() { exec->spin();}).detach();

    action_server_ = rclcpp_action::create_server<MultiChompOptimize>(
        this,
        "multi_chomp_optimize",
        std::bind(&ExtendedSpadesActionServer::handle_goal, this, _1, _2),
        std::bind(&ExtendedSpadesActionServer::handle_cancel,   this, _1),
        std::bind(&ExtendedSpadesActionServer::handle_accepted, this, _1));         

    RCLCPP_INFO(this->get_logger(), "Extended SPADES action server started");
}

rclcpp_action::GoalResponse
ExtendedSpadesActionServer::handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MultiChompOptimize::Goal> goal)
{
  // Forward check to optimizer’s parameters
  if (goal->num_robots == 0 ||
      goal->input_paths.size() != goal->num_robots) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting goal: num_robots = %u, paths.size() = %zu",
                goal->num_robots, goal->input_paths.size());
    return rclcpp_action::GoalResponse::REJECT;
  }

  if (optimizer_->get_num_robots() != static_cast<int>(goal->num_robots)) {
    RCLCPP_WARN(this->get_logger(),
                "Rejecting goal: goal num_robots %u != optimizer num_robots %d",
                goal->num_robots, optimizer_->get_num_robots());
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
ExtendedSpadesActionServer::handle_cancel(
    const std::shared_ptr<GoalHandleMultiChomp> /*goal_handle*/)
{
  RCLCPP_INFO(this->get_logger(), "Cancel request received");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ExtendedSpadesActionServer::handle_accepted(
    const std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
  std::thread(
      std::bind(&ExtendedSpadesActionServer::execute_goal, this, goal_handle)
  ).detach();
}

bool ExtendedSpadesActionServer::load_paths_into_state(
    const std::vector<nav_msgs::msg::Path> & paths)
{
  return optimizer_->set_paths(paths);
}

std::vector<nav_msgs::msg::Path>
ExtendedSpadesActionServer::export_state_to_paths(
    const std::vector<nav_msgs::msg::Path> & template_paths) const
{
  return optimizer_->get_paths(template_paths);
}

void ExtendedSpadesActionServer::execute_goal(
    const std::shared_ptr<GoalHandleMultiChomp> goal_handle)
{
  const auto goal = goal_handle->get_goal();

  MultiChompOptimize::Result result;
  MultiChompOptimize::Feedback feedback;

  if (!load_paths_into_state(goal->input_paths)) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to load input paths into optimizer");
    goal_handle->abort(std::make_shared<MultiChompOptimize::Result>(result));
    return;
  }

  const uint32_t max_iter =
      (goal->max_iterations > 0) ? goal->max_iterations : 100;

  for (uint32_t iter = 0; iter < max_iter; ++iter) {
    if (goal_handle->is_canceling()) {
      goal_handle->canceled(std::make_shared<MultiChompOptimize::Result>(result));
      RCLCPP_INFO(this->get_logger(), "Goal canceled");
      return;
    }


    optimizer_->solve_step();

    feedback.progress = static_cast<float>(iter + 1) /
                        static_cast<float>(max_iter);
    feedback.current_cost = 0.0f; // TODO: compute actual cost
    goal_handle->publish_feedback(
        std::make_shared<MultiChompOptimize::Feedback>(feedback));
  }

  result.optimized_paths = export_state_to_paths(goal->input_paths);
  goal_handle->succeed(std::make_shared<MultiChompOptimize::Result>(result));

  RCLCPP_INFO(this->get_logger(), "Goal succeeded");
}