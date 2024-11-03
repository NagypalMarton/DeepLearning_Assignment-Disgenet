# DeepLearning_Assignment-Disgenet
**Team name:** *CSA_NMP_TEAMS* <br>
**Team members' names and Neptun codes:** <br>
- Csibi Alexandra *(GPVFEV)*
- Nagypál Márton Péter *(Q88P3E)*

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**Functions of the files in the repository** <br>
- **ssh forder / ssh_config:**  contains the ssh settings for the machine made from the docker image
- **Dockerfile:** uses an official PyTorch image with CUDA support. It sets some environment variables and creates a working directory for the user. Installs the necessary packages (git, openssh-server, mc), creates a new user, then clones the GitHub repository to the working directory. Finally, it installs the Python dependencies from the requirements.txt file.
- **docker-compose.yml:** defines a service called srserver, which is built based on the local Dockerfile. The service image is deeplearningassdis, and the container name is deeplearningassdis_con. It appends the local directory to the container's user directory and opens the necessary ports: 8899 for Jupyter, 2299 for SSH, and 7860 for Gradio. The container will reboot unless it is shut down and also seizes an NVIDIA GPU to run.
- **requirements.txt:** lists the Python dependencies for the project, including scikit-learn, seaborn, pandas, numpy, matplotlib, torch-geometric, jupyterlab and related jupyter packages, and the tensorflow and gradio libraries. These packages are provided with different version numbers, ensuring compatibility and functionality needed for the project.
- **Mélytanulás_Beadandó_Csibi_Alexandra,_Nagypál_Márton.ipynb:** contains the Python code for the header solution, tagged
- **disgenet-GDA_cancer.csv & preprocessed_GDA_df_cancer.csv:**

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Dataset](https://www.disgenet.org/)
- [Related papers 1](https://arxiv.org/abs/1607.00653)
- [Related papers 2](https://arxiv.org/abs/1611.07308)

**How to run it (building and running the container, running your solution within the container)** <br>
XYZ
