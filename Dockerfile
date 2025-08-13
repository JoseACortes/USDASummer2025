FROM ubuntu:latest

# Update package list and install basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy current directory contents to container
COPY . /workspace

# Keep container running
CMD ["bash"]