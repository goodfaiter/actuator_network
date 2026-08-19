# Base image with CUDA 12.8 and cuDNN development libraries
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    software-properties-common \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create the project virtual environment outside the mounted workspace
RUN uv venv /opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.8 support
RUN uv pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu128

# Install opencode
RUN curl -fsSL https://opencode.ai/install | bash
ENV PATH="/root/.opencode/bin:$PATH"

# Setup ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list

# Install ROS2 Humble
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-argcomplete \
    ros-humble-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# Source ROS2 by default
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
