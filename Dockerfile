# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
FROM ubuntu:22.04

ARG GRADIO_SERVER_PORT=7860
ARG Jupiter_NoteBook_Port=8888
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Budapest

ENV HOME /home/testuser
RUN mkdir -p $HOME/DeepLearning_Assignment
WORKDIR $HOME/DeepLearning_Assignment
ENV DGN_GDA_cancer=${CSV:-$HOME/DeepLearning_Assignment/GDA_df_raw.csv}
ENV preProc_GDA_cancer=${CSV:-$HOME/DeepLearning_Assignment/GDA_df_processed.csv}
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# Expose port 8888 for Jupyter Notebook, 22 (ssh) && GRADIO
EXPOSE 22
EXPOSE 8888
EXPOSE ${GRADIO_SERVER_PORT}

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    mc \
    libgl1 \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    python3-pip \
    tzdata

SHELL ["/bin/bash", "-l", "-c"]

# Set timezone
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Add user
RUN useradd -rm -d $HOME -s /bin/bash -g root -G sudo -u 1000 testuser
RUN echo 'testuser:admin123' | chpasswd

# Clone the GitHub repository into the project directory
RUN git clone https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git $HOME/DeepLearning_Assignment-Disgenet

# Set the working directory to the cloned repository
WORKDIR $HOME/DeepLearning_Assignment-Disgenet

# Copy local files into the container
COPY . .
COPY /ssh/ssh_config /etc/ssh/ssh_config

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip3 install torch==2.1.1 --timeout=1000 \
    && pip3 install --no-cache-dir -r requirements.txt

# Start Jupyter lab with custom password
ENTRYPOINT service ssh start && jupyter-lab \
    --ip 0.0.0.0 \
    --port 8888 \
    --no-browser \
    --NotebookApp.notebook_dir='$home' \
    --ServerApp.terminado_settings="shell_command=['/bin/bash']" \
    --allow-root & \
    python -m gradio.app --server.port=${GRADIO_SERVER_PORT} --server.host=0.0.0.0