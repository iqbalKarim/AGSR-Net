import torch
import torch.nn as nn
import torch.nn.functional as F
from .my_config import device

class GAT(nn.Module):
    """
    A basic implementation of the GAT layer.

    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.
    """
    def __init__(self, in_features, out_features, activation=None, layer_norm=False, p=0):
        super(GAT, self).__init__()
        # Initialize the weights, bias, and attention parameters as
        # trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation

        self.layer_norm = layer_norm
        if layer_norm:
            self.norm_layer = nn.LayerNorm(out_features)

        self.dropout_layer = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, adj, input):
        """
        Forward pass of the GAT layer.

        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.

        Returns:
        Tensor: The output features of the nodes after applying the GAT layer.
        """
        ############# Your code here ############
        ## 1. Apply linear transformation and add bias
        H_k = input @ self.weight + self.bias
        ## 2. Compute the attention scores utilizing the previously
        ## established mechanism.
        ## Note: Keep in mind that employing matrix notation can
        ## optimize this process.
        N, D = H_k.shape
        H_k_1 = H_k @ self.phi[:D]
        H_k_2 = H_k @ self.phi[D:]
        S = H_k_1 + H_k_2.transpose(0, 1)
        ## Apply a non-linearity before masking
        S = F.leaky_relu(S)
        ## 3. Compute mask based on adjacency matrix
        mask = (adj + torch.eye(adj.shape[0]).to(adj.device)) == 0
        ## 4. Apply mask to the pre-attention matrix
        S_masked = torch.where(mask, torch.tensor(-9e15).to(adj.device), S)
        ## 5. Compute attention weights using softmax
        S_normalised = F.softmax(S_masked, dim=1)
        ## 6. Aggregate features based on attention weights
        ## Note: name the last line as `h`
        S_normalised = self.dropout_layer(S_normalised)
        h = S_normalised @ H_k
        ## (9-10 lines of code)
        #########################################
        h = self.norm_layer(h) if self.layer_norm else h
        return self.activation(h) if self.activation else h


class GCN(nn.Module):
    """
    A basic implementation of GCN layer.
    It aggregates information from a node's neighbors
    using mean aggregation.
    """
    def __init__(self, in_features, out_features, activation=None, layer_norm=False, p=0):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activation

        self.layer_norm = layer_norm
        if layer_norm:
            self.norm_layer = nn.LayerNorm(out_features)

        self.dropout_layer = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, x):
        """
        Forward pass of the GCN layer.

        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.

        Returns:
        Tensor: The output features of the nodes after applying the GCN layer.
        """
        # apply dropout
        x = self.dropout_layer(x)
        ############# Your code here ############
        ## Note:
        ## 1. Apply the linear transformation
        transformed = torch.matmul(x, self.weight)
        ## 2. Perform the graph convolution operation
        h = torch.matmul(adj, transformed) + self.bias
        ## Note: rename the last line as `output`
        ## (2 lines of code)
        #########################################
        h = self.norm_layer(h) if self.layer_norm else h
        h = self.activation(h) if self.activation else h
        return h


class GraphUnpool(nn.Module):
    '''
    Graph unpool layer
    '''
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]], device=device)
        new_X[idx] = X
        return A, new_X

class GraphPool(nn.Module):
    '''
    Graph polling layer
    '''
    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx

class GraphUnet(nn.Module):
    '''
    an improved version os Graph U-Net base a GAT
    '''
    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.start_gcn = GAT(in_dim, dim, nn.ReLU(True)).to(device)
        self.bottom_gcn = GAT(dim, dim, nn.ReLU(True)).to(device)
        self.end_gcn = GAT(2*dim, out_dim, nn.ReLU(True)).to(device)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GAT(dim, dim, nn.ReLU(True), p=0.5).to(device))
            self.up_gcns.append(GAT(dim, dim, nn.ReLU(True), p=0.5).to(device))
            self.pools.append(GraphPool(ks[i], dim).to(device))
            self.unpools.append(GraphUnpool().to(device))

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):

            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)

        return X, start_gcn_outs