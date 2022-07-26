import numpy as np

from ParamsNet import ParamsNet
from Dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset('../../dataset/dataset.csv', "../../dataset", 9000, 1)
    dataset.parse()
    net = ParamsNet(dataset, 1000)
    net.load('./models/phi')
    image, value = dataset.create_batch()
    for i in range(len(value)):
        x = net.forward(np.array([image[i]]))
        print(x, value[i])
        print((x - value[i])**2)