import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from torch import optim
from models.duckieNet import DuckieNet
from dataLoader.duckieSet import createDuckieLoader


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs):
    loss_history = []
    val_history = []
    for epoch in range(num_epochs):
        model.train() # Enter train mode
        d_loss_accum = 0
        phi_loss_accum = 0
        i_step = 0
        for obs, d, phi in train_loader:
            d = torch.reshape(d, (d.shape[0], 1))
            phi = torch.reshape(phi, (phi.shape[0], 1))
            obs_gpu = obs.float().cuda()
            d_gpu = d.float().cuda()
            phi_gpu = phi.float().cuda()
            obs_gpu = obs_gpu.requires_grad_(True)
            d_gpu = d_gpu.requires_grad_(True)
            phi_gpu = phi_gpu.requires_grad_(True)

            pred_d, pred_phi = model(obs_gpu)
            d_loss = loss(pred_d, d_gpu)
            phi_loss = loss(pred_phi, phi_gpu)
            optimizer.zero_grad()
            #d_loss.backward(retain_graph = True)
            #phi_loss.backward(retain_graph = True)
            loss_ = sum([d_loss, phi_loss])  # or loss = loss1 + loss2
            loss_.backward()
            optimizer.step()

            d_loss_accum += d_loss
            phi_loss_accum += phi_loss
            torch.cuda.empty_cache()

            del d, phi, obs, obs_gpu, d_gpu, d_loss, phi_gpu, phi_loss
            i_step += 1

        avg_d_loss = d_loss_accum / i_step
        avg_phi_loss = phi_loss_accum / i_step
        val = validate(model, val_loader)

        loss_history.append([avg_d_loss, avg_phi_loss])
        val_history.append(val)

        print(f'{epoch}, loss d={avg_d_loss} phi={avg_phi_loss}, len d={val[0]} phi={val[1]}')

    return loss_history, val_history


def validate(model, loader):
    model.eval()
    real_d = []
    real_phi = []
    pred_d = []
    pred_phi = []
    for i_step, (obs, d, phi) in enumerate(loader):

        obs = obs.float().cuda()
        pred_d1, pred_phi1 = model(obs)
        real_d += d
        real_phi += phi
        pred_d += pred_d1.cpu().data.numpy().tolist()
        pred_phi += pred_phi1.cpu().data.numpy().tolist()
        del pred_d1, pred_phi1

    d_len = np.abs(np.array(real_d)- np.array(pred_d).reshape(len(pred_d)))
    phi_len = np.abs(np.array(real_phi)- np.array(pred_phi).reshape(len(pred_phi)))
    return d_len.mean(), phi_len.mean()


def train():
    model = DuckieNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.L1Loss()
    train_loader, val_loader = createDuckieLoader(64)
    train_model(model.cuda(), train_loader, val_loader, loss, optimizer, 100)


if __name__ == '__main__':
    train()