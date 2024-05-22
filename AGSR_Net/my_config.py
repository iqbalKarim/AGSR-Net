import torch

USE_GPU = False
device = 'cuda' if torch.cuda.is_available() and USE_GPU else 'cpu'