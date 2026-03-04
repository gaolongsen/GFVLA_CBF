cmake_minimum_required(VERSION 3.8)
project(dualarmvla_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs_py REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR hardware/ros2
)

# Install executables
install(PROGRAMS
  hardware/ros2/dual_arm_ros2_bridge.py
  hardware/ros2/demo_tower_ros2.py
  hardware/ros2/demo_vla_letters_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  hardware/ros2/launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

