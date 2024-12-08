# DeepLearning_Assignment-Disgenet
**Team name:** *NMP_TEAM* <br>
**Team members' names and Neptun codes:** <br>
- Nagypál Márton Péter *(Q88P3E)*

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**functions of the files in the repository**<br>

## Functions of the Files in the Repository

- **`requirements.txt`**
  - Lists the dependencies required for the project, specifying the package names and versions.

- **`Dockerfile`**
  - Sets up a Docker container for the application.
  - Installs necessary dependencies, including Python, pip, and required Python packages.
  - Copies the application code into the container.
  - Exposes the port for the Gradio app and specifies the command to run the application.

- **`data.py`**
  - Interacts with the DisGeNET API to fetch disease-related data.
  - Handles API requests with rate-limiting.
  - Fetches disease IDs based on a specified disease type.
  - Downloads gene-disease association (GDA) data and saves it to a CSV file.

- **`datapreproc.py`**
  - Configures logging for data preprocessing.
  - Loads data from a CSV file into a pandas DataFrame.
  - Cleans and filters the data by selecting relevant columns and removing missing values.

- **`baseline_model.py`**
  - Configures logging for the baseline model.
  - Defines a Graph Convolutional Network (GCN) for link prediction.
  - Sets up the model architecture with GCN layers and a fully connected layer for link prediction.

Each file serves a specific purpose in the overall workflow of the project, from setting up the environment and fetching data to preprocessing and building a machine learning model.

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)

**how to run it (building and running the container, running your solution within the container)** <br>