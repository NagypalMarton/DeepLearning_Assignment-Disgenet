# Use an official PyTorch image with CUDA support 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG GRADIO_SERVER_PORT=7860

# Environment variable
ENV HOME /home/testuser
RUN mkdir -p $HOME/DeepLearning_Assignment
WORKDIR $HOME/DeepLearning_Assignment
ENV DGN_GDA_cancer=${CSV:-$HOME/DeepLearning_Assignment/disgenet-GDA_cancer.csv}
ENV preProc_GDA_cancer=${CSV:-$HOME/DeepLearning_Assignment/preprocessed_GDA_df_cancer.csv}
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# Install Git and other dependencies
RUN apt-get update && apt-get install -y git openssh-server mc libgl1 

SHELL ["/bin/bash", "-l", "-c"]

# ADD User
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
    && pip install --no-cache-dir -r requirements.txt

# Expose port 8888 for Jupyter Notebook, 22 (ssh) && GRADIO
EXPOSE 22
EXPOSE 8888
EXPOSE ${GRADIO_SERVER_PORT}

# Start Jupyter lab with custom password
ENTRYPOINT service ssh start && jupyter-lab \
	--ip 0.0.0.0 \
	--port 8888 \
	--no-browser \
	--NotebookApp.notebook_dir='$home' \
	--ServerApp.terminado_settings="shell_command=['/bin/bash']" \
	--allow-root & \
	python -m gradio.app --server.port=${GRADIO_SERVER_PORT} --server.host=0.0.0.0
	