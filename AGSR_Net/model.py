from .layers import *
from .ops import *
from .preprocessing import normalize_adj_torch
from .my_config import device


class AGSRNet(nn.Module):
    '''
    Original AGSRNet model class from the github repo https://github.com/basiralab/AGSR-Net
    '''

    def __init__(self, ks, args):
        super(AGSRNet, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.hr_dim).to(device)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim).to(device)
        self.gc1 = GCN(self.hr_dim, self.hidden_dim, nn.ReLU(True), True, p=0.5).to(device)
        self.gc2 = GCN(self.hidden_dim, self.hidden_dim, nn.ReLU(True), True, p=0.5).to(device)
        self.gc3 = GCN(self.hidden_dim, self.hr_dim, nn.ReLU(True)).to(device)

    def forward(self, lr, lr_dim, hr_dim):
        with torch.autograd.set_detect_anomaly(True):

            I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(device)
            A = normalize_adj_torch(lr).type(torch.FloatTensor).to(device)

            # Unet
            net_outs, start_gcn_outs = self.net(A, I)

            # GSRLayer
            adj, z = self.layer(A, net_outs)

            # 3 GCN layers
            z = self.gc1(adj, z)
            z = self.gc2(adj, z)
            z = self.gc3(adj, z)

            z = (z + z.t())/2
            z = z.fill_diagonal_(1)

        return torch.abs(z), net_outs, start_gcn_outs, adj


class Discriminator(nn.Module):
    '''
    GCN based discriminators
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.in_layer = GCN(args.hr_dim, 2 * args.hr_dim, nn.LeakyReLU(0.2, True)).to(device)
        self.hidden1 = GCN(2 * args.hr_dim, args.hr_dim, nn.LeakyReLU(0.2, True)).to(device)
        self.hidden2 = GCN(args.hr_dim, args.hr_dim // 2, nn.LeakyReLU(0.2, True)).to(device)
        self.out_layer = GCN(args.hr_dim // 2, 1, nn.Sigmoid()).to(device)

    def forward(self, inputs):
        np.random.seed(1)
        torch.manual_seed(1)

        A = normalize_adj_torch(inputs).to(device)
        # A = A + torch.eye(A.shape[0], dtype=torch.float, device=device)

        H = self.in_layer(A, inputs)
        H = self.hidden1(A, H)
        H = self.hidden2(A, H)
        output = self.out_layer(A, H)
        
        return output
    
def add_noise_to_adjacency_matrix(adj_matrix, noise_level=0.05):
    # Assume adj_matrix is a numpy array representing your adjacency matrix
    noise = np.random.randn(*adj_matrix.shape) * noise_level
    noisy_adj_matrix = adj_matrix + noise
    # Ensure the noisy adjacency matrix is still symmetric for undirected graphs
    noisy_adj_matrix = np.maximum(noisy_adj_matrix, 0)  # Remove negative values
    noisy_adj_matrix = (noisy_adj_matrix + noisy_adj_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(noisy_adj_matrix, 1)
    
    return noisy_adj_matrix

