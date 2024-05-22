import torch
import torch.nn as nn
from .initializations import *


class GSRLayer(nn.Module):
    '''
    GSRLayers which is used for predicting an HR connectome from the LR connectivity matrix and feature embeddings of the LR connectome
    '''

    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):

            lr = A
            lr_dim = lr.shape[0]
            f = X
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor).to(A.device)
            s_d = torch.cat((eye_mat, eye_mat), 0)

            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, f)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t())
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)
