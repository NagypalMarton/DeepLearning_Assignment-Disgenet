# DeepLearning_Assignment-Disgenet
**Team name:** *NMP_TEAMS* <br>
**Team members' names and Neptun codes:** <br>
- Nagypál Márton Péter *(Q88P3E)*

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**Functions of the files in the repository** <br>
- **Dockerfile:** uses an official PyTorch image with CUDA support. It sets some environment variables and creates a working directory for the user. Installs the necessary packages (git, openssh-server, mc), creates a new user, then clones the GitHub repository to the working directory. Finally, it installs the Python dependencies from the requirements.txt file.
- **docker-compose.yml:** defines a service called srserver, which is built based on the local Dockerfile. The service image is deeplearningassdis, and the container name is deeplearningassdis_con. It appends the local directory to the container's user directory and opens the necessary ports: 8899 for Jupyter, 2299 for SSH, and 7860 for Gradio. The container will reboot unless it is shut down and also seizes an NVIDIA GPU to run.
- **requirements.txt:** lists the Python dependencies for the project, including scikit-learn, seaborn, pandas, numpy, matplotlib, torch-geometric, jupyterlab and related jupyter packages, and the tensorflow and gradio libraries. These packages are provided with different version numbers, ensuring compatibility and functionality needed for the project.
- **disgenet-GDA_cancer.csv & preprocessed_GDA_df_cancer.csv:**

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)


**How to run it (building and running the container, running your solution within the container)** <br>
0. **Add the NVIDIA-Container-toolkit to OS**
Go to the [Official NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for installing and configure! And install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network).

1. **Build the Docker container:**
    ```bash
    docker-compose build
    ```

2. **Run the Docker container:**
    ```bash
    docker-compose up
    ```

3. **Access JupyterLab:**
    Open your web browser and navigate to `http://localhost:8899`. You should see the JupyterLab interface.

4. **Access the container via SSH:**
    ```bash
    ssh -p 2299 user@localhost
    ```

5. **Run the solution within the container:**
    Open the Jupyter notebook `Mélytanulás_Beadandó_Csibi_Alexandra,_Nagypál_Márton.ipynb` in JupyterLab and execute the cells to run the solution.
