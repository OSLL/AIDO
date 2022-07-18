import numpy as np
from PIL import Image
from time import time
from torch import Tensor

class Dataset:
    def __init__(self, filename: str, directory: str, batch_size: int, index: int):
        self.filename = filename
        self.directory = directory
        self.image_data = None
        self.value_data = None
        self.batch_size = batch_size
        self.current_batch_index = 0
        self.size = 75000
        self.index = index

    def parse(self):
        file = open(self.filename, "r")
        data = file.read().split('\n')
        parsed_data = []
        for i in data:
            parsed_data.append(list(map(float, i.split())))
        del data
        data = []
        for i in range(self.size):
            if len(parsed_data[i]) == 3:
                data.append([parsed_data[i][1], parsed_data[i][2]])
        self.value_data = np.array(data)
        data = []
        for i in range(self.size):
            print(i)
            data.append(np.asarray(Image.open('{}/{}.png'.format(self.directory, i)), dtype=np.float32))
        self.image_data = np.array(data)

    def create_batch(self):
        value_batch = self.value_data[self.current_batch_index:self.current_batch_index+self.batch_size:1]
        value_batch = np.array([value_batch[i][self.index] for i in range(len(value_batch))], dtype=np.float32)
        image_batch = self.image_data[self.current_batch_index:self.current_batch_index+self.batch_size:1]
        if self.current_batch_index + self.batch_size >= self.size:
            self.current_batch_index = 0
        else:
            self.current_batch_index += self.batch_size
        return image_batch, value_batch



if __name__ == "__main__":
    t = time()
    dataset = Dataset('../../dataset/dataset.csv', "../../dataset", 64, 0)
    dataset.parse()
    for i in range(5):
        dataset.create_batch()
    print(time() - t)