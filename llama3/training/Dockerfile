# # Use an NVIDIA CUDA base image that includes CUDA and cuDNN
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# # Install software-properties-common to add new repository
# RUN apt-get update && apt-get install -y software-properties-common

# # Add the deadsnakes PPA for newer Python versions
# RUN add-apt-repository ppa:deadsnakes/ppa

# # Install Python 3.9 and pip
# RUN apt-get update && apt-get install -y python3.9 python3.9-distutils python3.9-dev python3-pip && apt-get clean

# # Update alternatives to set python3.9 as the default Python version
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# # Upgrade pip to the latest version
# RUN python3.9 -m pip install --upgrade pip

# # Install PyTorch, torchvision, and torchaudio with GPU support
# RUN pip install --default-timeout=5000 torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118

# RUN apt install git

# RUN pip install git+https://github.com/huggingface/transformers@v4.31-release accelerate peft bitsandbytes trl tensorboardX huggingface_hub

# # Copy the training script
# COPY main.py /main.py

# # Run the training script
# CMD ["python", "/main.py"]

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
RUN pip install --default-timeout=5000 git+https://github.com/huggingface/transformers@v4.31-release accelerate peft bitsandbytes trl tensorboardX huggingface_hub

# Copy the training script
COPY main.py /main.py
COPY main2.py /main2.py
COPY run_scripts.sh /run_scripts.sh

RUN chmod +x /run_scripts.sh

CMD ["/run_scripts.sh"]