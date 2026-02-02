cmake_minimum_required(VERSION 3.0.2)
project(raven_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  src/video_stream_handler.py
  src/lane_segmentation.py
  src/lateral_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
