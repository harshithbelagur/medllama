# Use an NVIDIA CUDA base image that includes CUDA and cuDNN
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Install software-properties-common to add new repository
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    python3-pip \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to set python3.9 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip to the latest version
RUN python3.9 -m pip install --upgrade pip

# Install PyTorch, torchvision, and torchaudio with GPU support
RUN pip install --default-timeout=5000 torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# Install specific version of transformers and other packages
RUN pip install --default-timeout=5000 git+https://github.com/huggingface/transformers@v4.31-release accelerate bitsandbytes flask wandb tqdm datasets evaluate

# Make port 5000 available to the world outside this container
EXPOSE 5000

COPY inference.py /inference.py

# Run the inference script
CMD ["python", "/inference.py"]
