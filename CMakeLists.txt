cmake_minimum_required(VERSION 3.8)
project(extended_spades)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# 2. Define the Executable
add_executable(multi_chomp_node src/multi_chomp_node.cpp)

# 3. Link Dependencies
ament_target_dependencies(multi_chomp_node
  rclcpp
  rclcpp_action
  geometry_msgs
  visualization_msgs
  Eigen3
)

# 4. Install Rules (Where to put the binary)
install(TARGETS multi_chomp_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
