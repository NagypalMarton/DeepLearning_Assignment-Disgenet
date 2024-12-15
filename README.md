# DeepLearning_Assignment-Disgenet
**Team name:** *NMP_TEAM* <br>
**Team members' names and Neptun codes:** <br>
- Nagypál Márton Péter *(Q88P3E)*

**Project description** <br>
*Disease-gene interaction prediction with graph neural networks* <br>
The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, you'll apply deep learning to an important challenge of bioinformatics. By choosing this project, you'll gain experience in the intersection of deep learning and bioinformatics while extracting valuable insights from real-world data.

**functions of the files in the repository**<br>
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
- **`data forder`**
  -  Here you can find data downloaded from the Disgenet website in a CSV file, such as different cancer types and their characteristics, as well as existing gene-disease associations.
  -  **`disease_identifiers.csv`**
  -  **`gda_summary_data.csv`**

Each file serves a specific purpose in the overall workflow of the project, from setting up the environment and fetching data to preprocessing and building a machine learning model.

**Related works (papers, GitHub repositories, blog posts, etc)** <br>
- [Related GitHub repository](https://github.com/pyg-team/pytorch_geometric)
- [Related GitHub repository](https://github.com/sujitpal/pytorch-gnn-tutorial-odsc2021)
- [Related YouTube video](https://www.youtube.com/watch?v=-UjytpbqX4A&list=LL&index=1)
- [Dataset](https://www.disgenet.org/)

**how to run it (building and running the container, running your solution within the container)** <br>
Clone the Repository:
   1. clone the repository from GitHub:
   ```sh
   git clone [https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git](https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git)
   cd [https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git](https://github.com/NagypalMarton/DeepLearning_Assignment-Disgenet.git)
   ```
   2. Build the Docker Image
   ```
    docker build -t deeplearning_assignment .
   ```
   3. Run the Docker Container
   ```
    docker run -it deeplearning_assignment
   ```
**how to run the pipeline**

**how to train the models**

**how to evaluate the models**

---

**how to run the pipeline?**
1. **Start the Docker container**:
To run the project, build and start the environment using Docker:
```bash
docker build -t deeplearning_assignment .
docker run -it deeplearning_assignment
```
2. **Running the pipeline**:
Each step of the pipeline is encapsulated in a script. To start the pipeline, run:
```bash
python pipeline.py
```
This script automatically performs the data retrieval, preparation, and other steps.

---

**how to train the models?**
1. **Baseline model training**:
The baseline model is trained using the following script:
```bash
python train_baseline.py
```
This uses a simple Graph Convolutional Network (GCN) model for link prediction.

2. **GNN model training**:
To train the graph neural network, run:
```bash
python train_gnn.py
```
You can modify the configurations for training in the `config.py` file.

---

**how to evaluate the models?**
1. **Running the evaluation script**:
To evaluate the trained models, use the following script:
```bash
python evaluate_model.py
```
2. **Saving and visualizing results**:
The evaluation results (e.g. accuracy, ROC curves) are saved in the `results/` folder. The visualizations generated during the evaluation can be found in the `plots/` folder.
