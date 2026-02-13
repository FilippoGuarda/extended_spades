#include "rclcpp/rclcpp.hpp"
#include "extended_spades/extended_spades_action_server.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ExtendedSpadesActionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}