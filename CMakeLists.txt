cmake_minimum_required(VERSION 3.8)
project(extended_spades)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# add dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)        
find_package(rosidl_default_generators REQUIRED) 

# first we generate required interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MultiChompOptimize.action"
  DEPENDENCIES geometry_msgs nav_msgs
)

# create the executables ONCE INTERFACES ARE CREATED OR IT DOES NOT WORK
add_executable(multi_chomp
  src/multi_chomp.cpp
  src/extended_spades_action_server.cpp
  src/extended_spades_action_server_main.cpp
)

# linking libraries
ament_target_dependencies(multi_chomp
  rclcpp
  rclcpp_action
  geometry_msgs
  visualization_msgs
  nav_msgs
  Eigen3
  OpenCV
)

target_include_directories(multi_chomp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(multi_chomp ${OpenCV_LIBRARIES})

# link 
rosidl_target_interfaces(multi_chomp
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

install(TARGETS multi_chomp
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
