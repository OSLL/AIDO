from learning import BatchGenerator
from learning import ParamsNet
import torch
import numpy as np
from learning import BatchGenerator
import matplotlib.pyplot as plt



class Trainer:
    def __init__(self, model, model_path=None):
        self.batch = BatchGenerator(64)
        self.loss = []
        self.accuracy = []
        self.last = None
        self.best = 1
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model(self.device).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.criterion = torch.nn.MSELoss
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epoch, step_in_epoch, step_in_evaluate):
        self.model.train()
        for i in range(epoch):
            tmp = []
            for j in range(step_in_epoch):
                self.adam.zero_grad()
                obs, target = self.batch.create_batch()

                bs, h, w, c = obs.shape
                obs = obs.reshape(bs, c, h, w)
                output = self.model.forward(obs)
                target = torch.Tensor(target).to(self.device)
                loss = self.criterion()(output, target)
                loss.backward()
                tmp.append(loss.item())
                self.adam.step()
                self.last = loss.item()
            mean = np.array(tmp).mean()
            self.loss.append(mean)

            if self.last <= self.best:
                self.best = self.last
                torch.save(self.model.state_dict(), 'best.pkl')

            self.evaluate(step_in_evaluate)
        fig, ax = plt.subplots(1)
        x = np.arange(0, epoch, 1)
        ax.plot(x, self.loss, label='loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title("Train loss")
        plt.savefig('train.png')
        fig, ax = plt.subplots(1)
        x = np.arange(0, epoch, 1)

        ax.plot(x, self.accuracy, label='loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title("Eval loss")
        plt.savefig('eval.png')
        plt.show()


    def evaluate(self, eval_step):
        self.model.eval()
        loss = []
        for _ in range(eval_step):
            obs_batch, target = self.batch.create_batch()
            for i in range(len(obs_batch)):
                obs = obs_batch[i]
                obs = np.array([obs])
                bs, h, w, c = obs.shape
                obs = obs.reshape(bs, c, h, w)
                output = self.model.forward(obs)
                current = target[i]
                loss_d = self.criterion()(output, torch.Tensor(np.array([current])).to(self.device))
                loss.append(loss_d.item())
        self.accuracy.append(np.array(loss).mean())
        self.model.train()

if __name__ == "__main__":
    trainer = Trainer(ParamsNet)
    trainer.train(100, 10, 10)
