import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import joblib
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)

def load_data(file_path):
    """
    Load the data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Clean and filter the data by selecting relevant columns and removing missing values.

    Args:
        df (pandas.DataFrame): Raw data.

    Returns:
        pandas.DataFrame: Cleaned data with relevant columns and no missing values.
    """
    # Select columns of interest
    df = df[['diseaseId', 'geneId', 'score']]
    # Remove rows with any missing values
    df = df.dropna()
    return df


def encode_data(df):
    """
    Encode categorical variables (diseaseId and geneId) into numerical labels.

    Args:
        df (pandas.DataFrame): Cleaned data.

    Returns:
        tuple: 
            - pandas.DataFrame: DataFrame with encoded 'disease' and 'gene' columns.
            - LabelEncoder: Fitted encoder for diseases.
            - LabelEncoder: Fitted encoder for genes.
    """
    disease_encoder = LabelEncoder()
    gene_encoder = LabelEncoder()

    # Convert categorical IDs to numerical labels
    df['disease'] = disease_encoder.fit_transform(df['diseaseId'])
    df['gene'] = gene_encoder.fit_transform(df['geneId'])

    return df, disease_encoder, gene_encoder


def construct_graph(df, disease_encoder, gene_encoder):
    """
    Construct a graph data structure from the DataFrame.

    Args:
        df (pandas.DataFrame): Encoded data with 'disease' and 'gene' columns.
        disease_encoder (LabelEncoder): Encoder for disease IDs.
        gene_encoder (LabelEncoder): Encoder for gene IDs.

    Returns:
        torch_geometric.data.Data: Graph data containing node features, edge indices, and edge attributes.
    """
    # Calculate the number of diseases and genes
    num_diseases = len(disease_encoder.classes_)
    num_genes = len(gene_encoder.classes_)
    num_nodes = num_diseases + num_genes

    # Create edge index tensor (connect diseases to genes)
    edge_index = torch.tensor(
        [df['disease'].values, df['gene'].values + num_diseases],
        dtype=torch.long
    )

    # Create edge attributes (association scores)
    edge_attr = torch.tensor(df['score'].values, dtype=torch.float)

    # Normalize edge attributes to [0, 1]
    edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())

    # Create node features using node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.float)
    degrees.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    degrees.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

    # Enhance node features by adding node type information (0 for disease, 1 for gene)
    node_type = torch.cat([
        torch.zeros(num_diseases, dtype=torch.float),  # Disease nodes
        torch.ones(num_genes, dtype=torch.float)        # Gene nodes
    ]).unsqueeze(1)  # Shape: [num_nodes, 1]

    # Combine degrees and node type as features
    x = torch.cat([degrees.unsqueeze(1), node_type], dim=1)  # Shape: [num_nodes, 2]

    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def split_data(data):
    """
    Split the graph data into training, validation, and test sets using random link splitting.

    Args:
        data (torch_geometric.data.Data): Graph data.

    Returns:
        tuple: 
            - torch_geometric.data.Data: Training data.
            - torch_geometric.data.Data: Validation data.
            - torch_geometric.data.Data: Test data.
    """
    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)

    # Analyze class distribution
    analyze_class_distribution(train_data, 'Train')
    analyze_class_distribution(val_data, 'Validation')
    analyze_class_distribution(test_data, 'Test')

    return train_data, val_data, test_data


def analyze_class_distribution(data, split_name):
    """
    Log the number of positive and negative samples in a dataset split.

    Args:
        data (torch_geometric.data.Data): Data split.
        split_name (str): Name of the split (e.g., 'Train', 'Validation', 'Test').
    """
    num_positive = data.edge_label.sum().item()
    num_edges = data.edge_label.size(0)
    num_negative = num_edges - num_positive
    logging.info(f"{split_name} set - Positive samples: {int(num_positive)}, Negative samples: {int(num_negative)}")


def handle_class_imbalance(train_data):
    """
    Balance the training data by undersampling the majority class (negative samples).

    Args:
        train_data (torch_geometric.data.Data): Training data.

    Returns:
        torch_geometric.data.Data: Balanced training data.
    """
    # Count positive and negative samples
    num_positive = train_data.edge_label.sum().item()
    num_negative = train_data.edge_label.size(0) - num_positive

    logging.info(f"Original Train set - Positive: {int(num_positive)}, Negative: {int(num_negative)}")

    if num_positive == 0:
        logging.warning("No positive samples in training data.")
        return train_data

    # Calculate the number of negative samples to retain (equal to positive samples)
    num_neg_to_keep = int(num_positive)

    # Find indices of negative samples
    negative_indices = (train_data.edge_label == 0).nonzero(as_tuple=True)[0]
    if len(negative_indices) < num_neg_to_keep:
        logging.warning("Not enough negative samples to balance the training set.")
        num_neg_to_keep = len(negative_indices)

    # Randomly select negative samples to keep
    selected_neg_indices = negative_indices[torch.randperm(len(negative_indices))[:num_neg_to_keep]]

    # Find indices of positive samples
    positive_indices = (train_data.edge_label == 1).nonzero(as_tuple=True)[0]

    # Combine positive and selected negative indices
    balanced_indices = torch.cat([positive_indices, selected_neg_indices])

    # Update edge_index and edge_label
    train_data.edge_index = train_data.edge_index[:, balanced_indices]
    train_data.edge_label = train_data.edge_label[balanced_indices]

    logging.info(f"Balanced Train set - Positive: {int(train_data.edge_label.sum().item())}, Negative: {int(train_data.edge_label.size(0) - train_data.edge_label.sum().item())}")
    return train_data


# Main function to run data preparation
def main():
    try:
        output_dir = 'data'
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory '{output_dir}' is ready.")

        input_file = 'GDA_df_raw.csv'
        output_file = os.path.join(output_dir, 'processed_data.pt')

        # Load and clean data
        df = load_data(input_file)
        logging.info("Data loaded successfully.")

        df = clean_data(df)
        logging.info("Data cleaned.")

        # Encode data
        df, disease_encoder, gene_encoder = encode_data(df)
        logging.info("Data encoded.")

        # Save the label encoders
        joblib.dump(disease_encoder, os.path.join(output_dir, 'disease_encoder.pkl'))
        joblib.dump(gene_encoder, os.path.join(output_dir, 'gene_encoder.pkl'))
        logging.info("Label encoders saved.")

        # Construct graph
        data = construct_graph(df, disease_encoder, gene_encoder)
        logging.info("Graph constructed.")

        # Split data
        train_data, val_data, test_data = split_data(data)
        logging.info("Data split into train, validation, and test sets.")

        # Handle class imbalance in training data
        train_data = handle_class_imbalance(train_data)
        logging.info("Handled class imbalance in training data.")

        # Save processed data
        torch.save(train_data, os.path.join(output_dir, 'train_data.pt'))
        torch.save(val_data, os.path.join(output_dir, 'val_data.pt'))
        torch.save(test_data, os.path.join(output_dir, 'test_data.pt'))
        logging.info("Processed data saved.")

        logging.info("Data preprocessing completed and saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")


if __name__ == "__main__":
    main()