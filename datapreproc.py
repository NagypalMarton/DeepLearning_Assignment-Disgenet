# data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import joblib
import logging
import os

# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean and filter the data
def clean_data(df):
    # Filter relevant columns
    df = df[['diseaseId', 'geneId', 'score']]
    # Drop rows with missing values
    df = df.dropna()
    return df

# Encode categorical variables
def encode_data(df):
    disease_encoder = LabelEncoder()
    gene_encoder = LabelEncoder()

    df['disease'] = disease_encoder.fit_transform(df['diseaseId'])
    df['gene'] = gene_encoder.fit_transform(df['geneId'])

    return df, disease_encoder, gene_encoder

# Construct the graph
def construct_graph(df, disease_encoder, gene_encoder):
    # Calculate the number of diseases and genes
    num_diseases = len(disease_encoder.classes_)
    num_genes = len(gene_encoder.classes_)
    num_nodes = num_diseases + num_genes

    # Create edge index
    edge_index = torch.tensor(
        [df['disease'].values, df['gene'].values + num_diseases],
        dtype=torch.long
    )

    # Create edge attributes (scores)
    edge_attr = torch.tensor(df['score'].values, dtype=torch.float)

    # Normalize edge attributes
    edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())

    # Create node features using node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.float)
    degrees.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    degrees.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))

    # Convert degrees to feature matrix
    x = degrees.unsqueeze(1)  # Shape: [num_nodes, 1]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def split_data(data):
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
    num_positive = data.edge_label.sum().item()
    num_edges = data.edge_label.size(0)
    num_negative = num_edges - num_positive
    logging.info(f"{split_name} set - Positive samples: {num_positive}, Negative samples: {num_negative}")

# Main function to run data preparation
def main():
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)

    input_file = 'GDA_df_raw.csv'
    output_file = 'data/processed_data.pt'

    # Load and clean data
    df = load_data(input_file)
    df = clean_data(df)

    # Encode data
    df, disease_encoder, gene_encoder = encode_data(df)

    # Save the label encoders
    joblib.dump(disease_encoder, 'data/disease_encoder.pkl')
    joblib.dump(gene_encoder, 'data/gene_encoder.pkl')

    # Construct graph
    data = construct_graph(df, num_diseases, num_genes)

    # Split data
    train_data, val_data, test_data = split_data(data)


    # Save processed data
    torch.save(train_data, 'data/train_data.pt')
    torch.save(val_data, 'data/val_data.pt')
    torch.save(test_data, 'data/test_data.pt')
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    main()