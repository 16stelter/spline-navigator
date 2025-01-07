cmake_minimum_required(VERSION 3.8)
project(spline_navigator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(action_files
  "action/NavigatePath.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  DEPENDENCIES std_msgs builtin_interfaces geometry_msgs
)

install(
  PROGRAMS
    spline_navigator/navigator.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
