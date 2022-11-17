import numpy as np
from urllib import request
import gzip
import pickle
import os
from flow.tensor import Tensor, stack

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_idx = 0

        if self.shuffle:
            self.shuffle_idx = np.random.permutation(len(self.dataset))
        else:
            self.shuffle_idx = range(len(self.dataset))
    
    def __len__(self):
        if len(self.dataset) % self.batch_size == 0:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        i = self.batch_idx * self.batch_size
        if i < len(self.dataset):
            self.batch_idx += 1
            shuffle_idx = self.shuffle_idx[i:] if i + self.batch_size > len(self.dataset) else self.shuffle_idx[i: i + self.batch_size]
            data_batch = [self.dataset[idx] for idx in shuffle_idx]
            if isinstance(data_batch[0], tuple):
                return [stack(z) for z in zip(*data_batch)]
            else:
                return stack(data_batch)
        else:
            self.batch_idx = 0
            raise StopIteration

class Dataset:
    def __init__(self, path, train, transform=None, target_transform=None):
        self.path = path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        raise NotImplemented("must implement __len__ method of dataset.")

    def __iter__(self):
        return self
    
    def __getitem__(self, index):
        raise NotImplemented("must implement __getitem__ method of dataset.")


base_url = "http://yann.lecun.com/exdb/mnist/"

resources = [
    "train-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

class MNIST(Dataset):
    def __init__(self, path, train, download, transform=None, target_transform=None):
        super().__init__(path, train, transform, target_transform)
        if download:
            self.download()
        
        self.data_file = "train_mnist.pkl" if self.train else "test_mnist.pkl"
        self.data = self.load()
        
    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000
    
    def __getitem__(self, index):
        img, target = Tensor(self.data["images"][index]), Tensor(self.data["labels"][index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
        
    def check_exist(self):
        return (os.path.exists(os.path.join(self.path, "train_mnist.pkl")) and os.path.exists(os.path.join(self.path, "test_mnist.pkl")))
    
    def download(self):
        if self.check_exist():
            return
        for name in resources:
            print("Downloading " + name + "...")
            request.urlretrieve(base_url + name, os.path.join(self.path, name))
        print("Download complete.")
        self.save()
        print("Save complete.")

    def save(self):
        train_mnist = {}
        test_mnist = {}
        with gzip.open(os.path.join(self.path, resources[0]), 'rb') as f:
            train_mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16) / 255
        with gzip.open(os.path.join(self.path, resources[1]), 'rb') as f:
            test_mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16) / 255
        with gzip.open(os.path.join(self.path, resources[2]), 'rb') as f:
            train_mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(os.path.join(self.path, resources[3]), 'rb') as f:
            test_mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        # reshape
        train_mnist["images"] = train_mnist["images"].reshape(60000, 1, 28, 28)
        test_mnist["images"] = test_mnist["images"].reshape(10000, 1, 28, 28)
        # save
        with open(os.path.join(self.path, "train_mnist.pkl"), 'wb') as f:
            pickle.dump(train_mnist,f)
        with open(os.path.join(self.path, "test_mnist.pkl"), 'wb') as f:
            pickle.dump(test_mnist,f)

    def load(self):
        with open(os.path.join(self.path, self.data_file),'rb') as f:
            mnist = pickle.load(f)
        return mnist

if __name__ == "__main__":
    train_loader = DataLoader(
        MNIST('./data', train=True, download=True),
        batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape, target.shape)