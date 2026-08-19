#!/bin/bash

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Install the package in editable mode
cd /workspace
uv sync
uv pip install -e . --link-mode=copy

# Run the command passed to the container
exec "$@"
