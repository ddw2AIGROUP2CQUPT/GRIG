
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    BatchNorm


class GAT_3(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT_3, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=20)
        self.BatchNorm1 = BatchNorm(16 * 20)
        self.conv_linear1 = torch.nn.Linear(16 * 20, 16)
        self.BatchNorml1 = BatchNorm(16)

        self.conv2 = GATConv(16, 32, heads=16)
        self.BatchNorm2 = BatchNorm(32 * 16)
        self.conv_linear2 = torch.nn.Linear(32 * 16, 32)
        self.BatchNorml2 = BatchNorm(32)

        self.conv3 = GATConv(32, 48, heads=8)
        self.BatchNorm3 = BatchNorm(48 * 8)

        self.drop = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(48*8, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        batch = data.batch

        # block 1
        x = self.conv1(x, adj)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)

        # block2
        x = self.conv2(x, adj)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, adj)
        x = self.BatchNorm3(x)

        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.drop(x)
        x = self.linear(x)

        return x