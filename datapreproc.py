# data_preparation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data

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
    # Create edge index
    edge_index = torch.tensor([df['disease'].values, df['gene'].values + len(disease_encoder.classes_)], dtype=torch.long)

    # Create node features (one-hot encoding)
    num_diseases = len(disease_encoder.classes_)
    num_genes = len(gene_encoder.classes_)
    num_nodes = num_diseases + num_genes

    x = torch.eye(num_nodes)

    # Labels (using score as a proxy for association strength)
    y = torch.tensor(df['score'].values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Main function to run data preparation
def main():
    input_file = 'GDA_df_raw.csv'
    output_file = 'data/processed_data.pt'

    # Load and clean data
    df = load_data(input_file)
    df = clean_data(df)

    # Encode data
    df, disease_encoder, gene_encoder = encode_data(df)

    # Construct graph
    data = construct_graph(df, disease_encoder, gene_encoder)

    # Save processed data
    torch.save(data, output_file)
    print(f"Data preprocessing completed and saved to {output_file}")

if __name__ == "__main__":
    main()