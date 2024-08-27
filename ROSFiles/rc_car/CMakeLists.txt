cmake_minimum_required(VERSION 3.8)
project(rc_car)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
install(DIRECTORY
src
urdf
meshes
launch
worlds
competition
config
DESTINATION share/${PROJECT_NAME}/
)

install(
    DIRECTORY include/rc_car
    DESTINATION include
    )
    
install(
    PROGRAMS
    launch/robot_description_publisher.py
    src/keyboard_control_node.py
    src/rl_controller.py
    DESTINATION lib/${PROJECT_NAME}
    )
    
find_package(robot_state_publisher REQUIRED)
find_package(xacro REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ros2_control REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
