import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    BatchNorm


class GCN_8_plus(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim = 16, inithead = 16):
        super(GCN_8_plus, self).__init__()
        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=3)
        self.BatchNorm1 = BatchNorm(initdim * inithead)
        self.conv_linear1 = torch.nn.Linear(initdim * inithead, initdim)
        self.BatchNorml1 = BatchNorm(initdim)

        self.conv2 = GATConv(initdim, initdim * 2, heads = int(inithead / 2), edge_dim=3)
        self.BatchNorm2 = BatchNorm(initdim * inithead)
        self.conv_linear2 = torch.nn.Linear(initdim * inithead, initdim * 2)
        self.BatchNorml2 = BatchNorm(initdim * 2)

        self.conv3 = GATConv(initdim * 2, initdim * 4, heads = int(inithead / 4), edge_dim=3) 
        self.BatchNorm3 = BatchNorm(initdim * inithead)

        # self.drop = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(initdim * inithead, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # block 1
        x = self.conv1(x, adj, edge_attr)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)
        # block2

        x = self.conv2(x, adj, edge_attr)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, adj, edge_attr)
        x = self.BatchNorm3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        # x = self.drop(x)
        x = self.linear(x)

        return x


class GCN_Layer_4(torch.nn.Module):
    def __init__(self, num_features, num_classes, initdim = 16, inithead = 16):
        super(GCN_Layer_4, self).__init__()
        self.conv1 = GATConv(num_features, initdim, heads=inithead, edge_dim=3)
        self.BatchNorm1 = BatchNorm(initdim * inithead)
        self.conv_linear1 = torch.nn.Linear(initdim * inithead, initdim)
        self.BatchNorml1 = BatchNorm(initdim)

        self.conv2 = GATConv(initdim, initdim * 2, heads = int(inithead / 2), edge_dim=3)
        self.BatchNorm2 = BatchNorm(initdim * inithead)
        self.conv_linear2 = torch.nn.Linear(initdim * inithead, initdim * 2)
        self.BatchNorml2 = BatchNorm(initdim * 2)

        self.conv3 = GATConv(initdim * 2, initdim * 4, heads = int(inithead / 4), edge_dim=3) 
        self.BatchNorm3 = BatchNorm(initdim * inithead)
        self.conv_linear3 = torch.nn.Linear(initdim * inithead, initdim * 4)
        self.BatchNorml3 = BatchNorm(initdim * 4)

        self.conv4 = GATConv(initdim * 4, initdim * 8, heads = int(inithead / 8), edge_dim=3) 
        self.BatchNorm4 = BatchNorm(initdim * inithead)

        # self.drop = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(initdim * inithead, num_classes)

    def forward(self, data):
        x = data.x

        adj = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        # print("batch",data.batch.shape)
        # edge_attr = data.edge_attr
        # x, att1 = self.conv1(x, adj, return_attention_weights=True)

        # block 1
        x = self.conv1(x, adj, edge_attr)
        x = self.BatchNorm1(x)
        x = F.relu(x)
        x = self.conv_linear1(x)
        x = self.BatchNorml1(x)
        x = F.relu(x)

        # block2
        x = self.conv2(x, adj, edge_attr)
        x = self.BatchNorm2(x)
        x = F.relu(x)
        x = self.conv_linear2(x)
        x = self.BatchNorml2(x)
        x = F.relu(x)

        # block 3
        x = self.conv3(x, adj, edge_attr)
        x = self.BatchNorm3(x)
        x = F.relu(x)
        x = self.conv_linear3(x)
        x = self.BatchNorml3(x)
        x = F.relu(x)

        # block 4
        x = self.conv4(x, adj, edge_attr)
        x = self.BatchNorm4(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        # x = self.drop(x)
        x = self.linear(x)

        return x


class GCN_block(torch.nn.Module):
    def __init__(self, input_dims, output_dims, head_nums, do_linear=True,linear_outdims=None):
        super(GCN_block, self).__init__()

        self.do_linear=do_linear
        self.conv0 = GATConv(input_dims,output_dims,heads=head_nums,edge_dim=3)
        self.BN0 = BatchNorm(output_dims*head_nums)
        self.relu = torch.nn.ReLU()
        if self.do_linear:
            self.linear = torch.nn.Linear(output_dims*head_nums,linear_outdims)
            self.BN1 = BatchNorm(linear_outdims)



    def forward(self, x, adj,edge_attr):

        x = self.conv0(x, adj,edge_attr=edge_attr)
        x = self.BN0(x)
        x = self.relu(x)

        if self.do_linear:
            x = self.linear(x)

            x = self.BN1(x)
            x = self.relu(x)

        return x
        
        
class GCN(torch.nn.Module):
    def __init__(self,num_features,num_classes,init_out_dim=16,init_head_num=48):
        super(GCN, self).__init__()

        self.block1 = GCN_block(num_features,init_out_dim,init_head_num,linear_outdims=init_out_dim)    # 10 ->16

        self.block2 = GCN_block(init_out_dim,init_out_dim * 2,int(init_head_num/2),linear_outdims=init_out_dim*2)    # 16 ->32

        self.block3 = GCN_block(init_out_dim * 2, init_out_dim * 4, int(init_head_num / 4),linear_outdims=init_out_dim * 4)  # 32 ->64

        self.block4 = GCN_block(init_out_dim * 4, init_out_dim * 8, int(init_head_num / 8),linear_outdims=init_out_dim * 8)  # 64 -> 128

        self.block5 = GCN_block(init_out_dim * 8, init_out_dim * 16, int(init_head_num / 16),do_linear=False)  #128 -> 256

        self.head = torch.nn.Linear(init_out_dim *init_head_num, num_classes)
        
    def forward(self,data):
        x = data.x

        adj = data.edge_index
        edge_attr =data.edge_attr

        batch = data.batch

        x = self.block1(x,adj,edge_attr)
        x = self.block2(x, adj,edge_attr)
        x = self.block3(x, adj,edge_attr)
        x = self.block4(x, adj,edge_attr)
        x = self.block5(x, adj,edge_attr)

        x = global_mean_pool(x,batch)
        x = self.head(x)
        return x