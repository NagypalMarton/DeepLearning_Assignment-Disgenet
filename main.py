import gradio as gr
import torch
import torch_geometric as tg
from model import GCNLinkPredictor  # Import your GNN model
import numpy as np

# Load your pre-trained model
model = GCNLinkPredictor(input_dim=10, hidden_dim=32)  # Adjust input_dim and hidden_dim as needed
model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval()

# Define a function to make predictions
def predict(input_data):
    # Convert input data to tensor and reshape as needed
    input_data = np.array(input_data).reshape(1, -1)
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Create dummy edge_index and edge_label_index for prediction
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Dummy edge index
    edge_label_index = torch.tensor([[0], [1]], dtype=torch.long)  # Dummy edge label index

    with torch.no_grad():
        prediction = model(input_tensor, edge_index, edge_label_index)
    return prediction.numpy()[0]

# Create a Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter input data here..."),
    outputs="text",
    title="Deep Learning Model Prediction",
    description="Enter the input data to get predictions from the deep learning model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()