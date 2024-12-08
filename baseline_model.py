import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("baseline_model.log"),
        logging.StreamHandler()
    ]
)

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GCNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)  # Concatenate source and target node embeddings

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Extract source and target node embeddings
        src, dst = edge_index
        z_src = z[src]
        z_dst = z[dst]
        # Concatenate source and target embeddings
        z_cat = torch.cat([z_src, z_dst], dim=1)
        # Compute logits
        logits = self.fc(z_cat).squeeze()
        return torch.sigmoid(logits)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_index)

def load_data(file_path):
    """
    Load the processed PyTorch Geometric Data object.

    Args:
        file_path (str): Path to the .pt file.

    Returns:
        torch_geometric.data.Data: Loaded data.
    """
    data = torch.load(file_path)
    return data

def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        preds = (out >= 0.5).int()
        labels = data.edge_label.int()
        acc = accuracy_score(labels.cpu(), preds.cpu())
        report = classification_report(labels.cpu(), preds.cpu(), target_names=['Negative', 'Positive'])
        cm = confusion_matrix(labels.cpu(), preds.cpu())
    return acc, report, cm

def main():
    try:
        # File paths
        train_file = 'data/train_data.pt'
        val_file = 'data/val_data.pt'
        test_file = 'data/test_data.pt'
        model_save_path = 'models/baseline_pyg_model.pth'

        # Load data
        logging.info("Loading training data...")
        train_data = load_data(train_file)
        logging.info("Loading validation data...")
        val_data = load_data(val_file)
        logging.info("Loading test data...")
        test_data = load_data(test_file)

        # Initialize model
        num_node_features = train_data.x.size(1)  # Number of node features
        hidden_dim = 64
        model = GCNLinkPredictor(num_node_features, hidden_dim)

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

        # Define optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        # Training loop
        epochs = 100
        best_val_acc = 0
        os.makedirs('models', exist_ok=True)

        logging.info("Starting training...")

        for epoch in range(1, epochs + 1):
            loss = train(model, optimizer, criterion, train_data)
            val_acc, val_report, val_cm = evaluate(model, val_data)

            logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                logging.info(f'Best model saved with Val Acc: {best_val_acc:.4f}')

        # Load the best model for testing
        model.load_state_dict(torch.load(model_save_path))
        test_acc, test_report, test_cm = evaluate(model, test_data)

        logging.info("=== Test Evaluation ===")
        logging.info(f"Test Accuracy: {test_acc:.4f}\n")
        logging.info("Classification Report:\n" + test_report)
        logging.info("Confusion Matrix:\n" + str(test_cm))

    except Exception as e:
        logging.error(f"An error occurred during baseline model training: {e}")

if __name__ == "__main__":
    main()