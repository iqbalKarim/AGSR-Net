from torch.optim.lr_scheduler import StepLR
from .preprocessing import *
from .model import *
import torch.optim as optim
from tqdm import tqdm
from .my_config import device


def train(model, subjects_adj, subjects_labels, test_lr, test_hr, args):
    optimizerG = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    schedulerG = StepLR(optimizerG, step_size=40, gamma=0.8)
    criterion = nn.MSELoss()

    with tqdm(range(args.epochs), desc='Training') as tepoch:
        for epoch in tepoch:
            epoch_loss = []
            epoch_error = []
            with torch.autograd.set_detect_anomaly(True):
                model.train()
                for lr, hr in zip(subjects_adj, subjects_labels):
                    optimizerG.zero_grad()

                    # augmentation through noise injection
                    hr = add_noise_to_adjacency_matrix(hr, 0.1)
                    lr = add_noise_to_adjacency_matrix(lr, 0.1)

                    hr = pad_HR_adj(hr, args.padding)
                    lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                    padded_hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                    eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                    model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                        lr, args.lr_dim, args.hr_dim)

                    recon_loss = criterion(model_outputs, padded_hr)

                    mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                        model.layer.weights, U_hr) + recon_loss

                    mse_loss.backward()
                    optimizerG.step()

                    epoch_loss.append(mse_loss.item())
                    epoch_error.append(recon_loss.item())

            schedulerG.step()
            
            if (epoch % 10 == 0) or (epoch == args.epochs - 1):
                epoch_test_loss = []
                epoch_test_error = []
                model.eval()
                with torch.no_grad():
                    for lr, hr in zip(test_lr, test_hr):
                        hr = pad_HR_adj(hr, args.padding)
                        lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                        padded_hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                        eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')
                        model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                            lr, args.lr_dim, args.hr_dim)
                        recon_loss = criterion(model_outputs, padded_hr)
                        mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                            model.layer.weights, U_hr) + recon_loss
                        
                        epoch_test_loss.append(mse_loss.item())
                        epoch_test_error.append(recon_loss.item())

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(total_loss=np.mean(epoch_loss),
                               mse_error=f'{np.mean(epoch_error) * 100:.2f}%', 
                               total_test_loss=np.mean(epoch_test_loss),
                               mse_test_error=f'{np.mean(epoch_test_error) * 100:.2f}%'
                               )
            print()


def test(model, test_adj, args):

    preds_list = []

    for lr in test_adj:
        all_zeros_lr = not np.any(lr)
        if all_zeros_lr == False :
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            preds, a, b, c = model(lr, args.lr_dim, args.hr_dim)
            preds = unpad(preds, args.padding).detach().cpu().numpy()
            preds_list.append(preds)

    return np.stack(preds_list)
            
