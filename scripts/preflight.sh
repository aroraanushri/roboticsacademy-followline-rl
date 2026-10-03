#!/usr/bin/env bash
set -u
# Diagnostics only: run inside the SAME container that runs the exercise.
printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unset}"
printf 'ROS_DOMAIN_ID=%s\n' "${ROS_DOMAIN_ID:-0}"
printf 'GZ_PARTITION=%s\n' "${GZ_PARTITION:-unset}"
command -v ros2 || exit 1
command -v gz || exit 1
ros2 topic list -t
gz service -l
python -c 'import rclpy, cv2, numpy, gymnasium, cv_bridge; print("Python imports OK")'
