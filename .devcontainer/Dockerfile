# Base: Official PyTorch with CUDA 12.6
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Avoid debconf errors
ENV DEBIAN_FRONTEND=noninteractive
ENV SDL_VIDEODRIVER=x11
ENV XAUTHORITY=/tmp/.Xauthority

# Install system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    swig \
    && apt-get clean

# Upgrade pip and install packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
