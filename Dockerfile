# Use the official Ubuntu 22.04 as a base image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Update the package list and install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install the required Python packages
RUN pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu # Install PyTorch cpu version
RUN pip3 install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Gradio app will run on
EXPOSE 7860

# Command to run the Gradio web app
#CMD ["python3", "app.py"]
CMD ["python3", "baseline_model.py"]