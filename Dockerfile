# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Clone your GitHub repository into the container
RUN git clone https://github.com/your-username/DeepLearning_Assignment-Disgenet.git

# Set the working directory to the cloned repository
WORKDIR /app/DeepLearning_Assignment-Disgenet

# Run a script or open a bash shell when the container starts
CMD ["bash"]
