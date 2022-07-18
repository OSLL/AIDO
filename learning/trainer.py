from learning import Batch
from learning import ParamsNet
import torch


class Trainer:
    def __init__(self, name, size):
        self.net = ParamsNet()
        self.directory = name
        self.batch = Batch(size, name)
        self.adam = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.device = torch.device('cuda:0')
        self.log_f = open(f'./{self.directory}.csv', "w+")
        self.loss_f = torch.nn.MSELoss

    def loss(self, pred, train):
        L2_1 = ((pred[::, 0:1:1] - train[::, 0:1:1]) ** 2).mean()
        L2_2 = ((pred[::, 1::1] - train[::, 1::1]) ** 2).mean()
        L2 = L2_1 + L2_2
        return L2.mean()

    def train(self, epoch):
        for i in range(epoch):
            print(f'epoch = {i}')
            self.adam.zero_grad()
            data, train = self.batch.create_batch()
            train = torch.from_numpy(train).to(self.device)
            prediction = self.net.forward(data)
            print(prediction)
            loss = self.loss(prediction, train)
            loss.backward()
            self.adam.step()
            del data, train, prediction
            if i % 5000 == 0 and i != 0:
                self.net.eval()
                self.evaluate(i)
                self.save(i)
                self.net.train()
        self.log_f.close()

    def save(self, name):
        self.net.save(f'./{self.directory}/{name}')

    def load(self, path):
        self.net.load(path)

    def evaluate(self, epoch_num):
        print("EVALUATE")
        data, train = self.batch.create_batch()
        train = torch.from_numpy(train).to(self.device)
        prediction = self.net.forward(data)
        loss = self.loss(prediction, train)
        print("print file")
        print('{}, {}, {}\n'.format(epoch_num, loss, (1 - loss / 10) * 100))
        self.log_f.write('{}, {}, {}\n'.format(epoch_num, loss, (1 - loss / 10) * 100))

    def local_evaluate(self):
        for _ in range(100):
            data, train = self.batch.create_batch()
            train = torch.from_numpy(train).to(self.device)
            prediction = self.net.forward(data)
            loss = self.loss(prediction, train)
            print(f'pred = {prediction.item()}')
            print(f'real = {train.item()}')
            print(f'loss = {loss}')


if __name__ == "__main__":
    train = Trainer('dist', 1)
    train.load('./dist/950')
    train.local_evaluate()
