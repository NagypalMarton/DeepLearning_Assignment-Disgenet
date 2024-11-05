# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ULFCdhBgLb3WP-Xnx80CVQmzGk0jimMU
"""

import torch
import torch.nn.functional as F
import torch_geometric as tg
import pytorch_lightning as pl
from torchmetrics.classification import AUROC

class GCNLinkPredictor(pl.LightningModule):
  def __init__(self, input_dim, hidden_dim, lr=1e-2):
    super().__init__()
    self.save_hyperparameters()

    # model architecture
    self.conv1 = tg.nn.GCNConv(input_dim, hidden_dim)
    self.conv2 = tg.nn.GCNConv(hidden_dim, hidden_dim)
    self.lr = lr

    # metrics
    self.train_auroc = AUROC(task="binary")
    self.val_auroc = AUROC(task="binary")
    self.test_auroc = AUROC(task="binary")

  def forward(self, x, edge_index, edge_label_index):
    x = F.relu(self.conv1(x, edge_index))
    x = F.relu(self.conv2(x, edge_index))

    # Get node embeddings for each node in the edge pairs
    src_nodes  = x[edge_label_index[0]]
    dst_nodes  = x[edge_label_index[1]]

    link_logits = torch.sum(src_nodes * dst_nodes, dim=-1)

    return link_logits

  def _shared_step(self, batch, batch_idx, stage):
    x, edge_index = batch.x, batch.edge_index
    edge_label_index = batch.edge_label_index
    edge_labels = batch.edge_label

    # Forward pass to get link logits
    link_logits = self.forward(x, edge_index, edge_label_index)

    loss = F.binary_cross_entropy_with_logits(link_logits, edge_labels.float())

    # Convert logits to binary predictions for metrics
    preds = torch.sigmoid(link_logits) >= 0.4

    # Log metrics based on the current stage (train, val, test)
    if stage == 'train':
      self.train_auroc(link_logits, edge_labels)
      self.log('train_loss', loss)
      self.log('train_auroc', self.train_auroc, prog_bar=True)

    elif stage == 'val':
      self.val_auroc(link_logits, edge_labels)
      self.log('val_loss', loss)
      self.log('val_auroc', self.val_auroc, prog_bar=True)

    elif stage == 'test':
      self.test_auroc(link_logits, edge_labels)
      self.log('test_loss', loss)
      self.log('test_auroc', self.test_auroc, prog_bar=True)

    return loss


  def training_step(self, batch, batch_idx):
    return self._shared_step(batch, batch_idx, stage='train')

  def validation_step(self, batch, batch_idx):
    return self._shared_step(batch, batch_idx, stage='val')

  def test_step(self, batch, batch_idx):
    return self._shared_step(batch, batch_idx, stage='test')

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)