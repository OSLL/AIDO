import torch
import numpy as np
from ml.batchGenerator import BatchGenerator
import matplotlib.pyplot as plt
from ml_model import ParallelNet


class Trainer:
    def __init__(self, model, model_path=None):
        self.batch = BatchGenerator(64)
        self.d_train_loss = []
        self.phi_train_loss = []
        self.accuracy_d = []
        self.accuracy_phi = []
        self.last_d = None
        self.last_phi = None
        self.best_phi = 1
        self.best_d = 1
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model(self.device).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.criterion = torch.nn.MSELoss
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epoch, step_in_epoch, step_in_evaluate):
        self.model.train()
        for i in range(epoch):
            d_tmp = []
            phi_tmp = []
            for j in range(step_in_epoch):
                self.adam.zero_grad()
                obs, d, phi = self.batch.create_batch()
                bs, h, w, c = obs.shape
                obs = obs.reshape(bs, c, h, w)
                pred_d, pred_phi = self.model.forward(obs)
                ds = d.shape[0]
                d = np.array([d]).reshape(ds, 1)
                d = torch.Tensor(d).to(self.device)

                phis = phi.shape[0]
                phi = np.array([phi]).reshape(phis, 1)
                phi = torch.Tensor(phi).to(self.device)
                d_loss = self.criterion()(pred_d, d)
                phi_loss = self.criterion()(pred_phi, phi)
                d_loss.backward()
                self.adam.step()
                self.adam.zero_grad()
                phi_loss.backward()
                self.adam.step()
                phi_tmp.append(phi_loss.item())
                d_tmp.append(d_loss.item())
                self.last_d = d_loss.item()
                self.last_phi = phi_loss.item()
            mean_phi = np.array(phi_tmp).mean()
            self.phi_train_loss.append(mean_phi)
            mean_d = np.array(d_tmp).mean()
            self.d_train_loss.append(mean_d)

            if self.last_d <= self.best_d and self.last_phi <= self.best_phi:
                self.best_d = self.last_d
                self.best_phi = self.last_phi
                torch.save(self.model.state_dict(), 'best.pkl')
            elif self.last_d > self.best_d and self.last_phi < self.best_phi:
                self.best_phi = self.last_phi
                torch.save(self.model.state_dict(), 'best_phi.pkl')
            elif self.last_d <= self.best_d and self.last_phi > self.best_phi:
                self.best_d = self.last_d
                torch.save(self.model.state_dict(), 'best_d.pkl')
            self.evaluate(step_in_evaluate)
        fig, ax = plt.subplots(1)
        x = np.arange(0, epoch, 1)
        ax.plot(x, self.phi_train_loss, label='loss phi')
        ax.plot(x, self.d_train_loss, label='loss d')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Train loss")
        plt.savefig('train.png')
        fig, ax = plt.subplots(1)
        x = np.arange(0, epoch, 1)

        ax.plot(x, self.accuracy_d, label='loss phi')
        ax.plot(x, self.accuracy_phi, label='loss d')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title("Eval loss")
        plt.savefig('eval.png')
        plt.show()

    def evaluate(self, eval_step):
        self.model.eval()
        d = []
        phi = []
        for _ in range(eval_step):
            obs_batch, d_batch, phi_batch = self.batch.create_batch()
            for i in range(len(obs_batch)):
                obs = obs_batch[i]
                obs = np.array([obs])
                bs, h, w, c = obs.shape
                obs = obs.reshape(bs, c, h, w)
                pred_d, pred_phi = self.model.forward(obs)
                current_d = d_batch[i]
                current_phi = phi_batch[i]
                loss_d = self.criterion()(pred_d, torch.Tensor(np.array([current_d])).to(self.device))
                d.append(loss_d.item())
                loss_phi = self.criterion()(pred_phi, torch.Tensor(np.array([current_phi])).to(self.device))
                phi.append(loss_phi.item())
        self.accuracy_d.append(np.array(d).mean())
        self.accuracy_phi.append(np.array(phi).mean())
        self.model.train()


    def local_eval(self, eval_step):
        for i in range(eval_step):
            self.evaluate(1)
        fig, ax = plt.subplots(1)
        x = np.arange(0, eval_step, 1)

        ax.plot(x, self.accuracy_d, label='loss phi')
        ax.plot(x, self.accuracy_phi, label='loss d')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title("Eval loss")
        plt.savefig('eval.png')
        plt.show()


if __name__ == "__main__":
    trainer = Trainer(ParallelNet, './best.pkl')
    trainer.local_eval(10)
