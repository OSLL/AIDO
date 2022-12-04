import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader


class DuckieSet(Dataset):
    def __init__(self, path_to_image, path_to_markup, transform=None):
        self.transform = transform
        markup_tmp = np.load(path_to_markup)
        self.markup = [[int(i[0]), float(i[1]), float(i[2])] for i in markup_tmp]
        self.path_to_image = path_to_image

    def __len__(self):
        return len(self.markup)

    def __getitem__(self, index):
        img = cv2.imread(f'{self.path_to_image}/{self.markup[index][0]}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']
        img = img.reshape((3, 64, 64))
        return img, self.markup[index][1], self.markup[index][2]


def createDuckieLoader(batch_size):
    train_dataset = DuckieSet('./train_dataset', './train_markup.npy',
                              transform=A.Compose([
                                  A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=0.5),
                              ])
                              )
    data_size = len(train_dataset)
    validation_fraction = .2

    val_split = int(np.floor((validation_fraction) * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    val_indices, train_indices = indices[:val_split], indices[val_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size,
                                             sampler=val_sampler)
    return train_loader, val_loader
