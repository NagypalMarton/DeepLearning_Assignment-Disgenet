# Use an official PyTorch image with CUDA support 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Environment variable
ENV HOME /home/testuser
RUN mkdir -p $HOME/DeepLearning_Assignment
WORKDIR $HOME/DeepLearning_Assignment
ENV CSV=${CSV:-$HOME/DeepLearning_Assignment/disgenet-GDA.csv}

# Install Git and other dependencies
RUN apt-get update && apt-get install -y git

# Copy local files into the container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Clone the GitHub repository into the project directory
RUN git clone https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git $HOME/DeepLearning_Assignment-Disgenet

# Set the working directory to the cloned repository
WORKDIR $HOME/DeepLearning_Assignment-Disgenet

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Default command to run JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
