import numpy as np
from urllib import request
import gzip
import pickle

class DataLoader:
    def __init__(self, dataset, batchsize, shuffle):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.shuffle_idx = None
        if self.shuffle:
            shuffle_idx = np.random.permutation(len(self.dataset))
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # TODO get batchsize data from dataset using shuffle_idx
        pass


class Dataset:
    def __init__(self, path, train, download):
        self.path = path
        self.train = train
        self.download = download
    
    def __len__(self):
        raise NotImplemented("must implement __len__ method of dataset.")

    def __iter__(self):
        return self
    
    def __next__(self):
        # TODO get one data from dataset
        raise NotImplemented("must implement __next__ method of dataset.")

filename = [
    ["training_images","train-images-idx3-ubyte.gz"],
    ["test_images","t10k-images-idx3-ubyte.gz"],
    ["training_labels","train-labels-idx1-ubyte.gz"],
    ["test_labels","t10k-labels-idx1-ubyte.gz"]
]

class MNIST(Dataset):
    def __init__(self, path, train, download):
        super().__init__(path, train, download)
        if self.download:
            self.download_data()
            self.save()
        else:
            self.load()
    
    def __len__(self):
        # TODO
        pass
    
    def __next__(self):
        # TODO
        pass

    def download_data(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in filename:
            print("Downloading "+name[1]+"...")
            request.urlretrieve(base_url+name[1], name[1])
        print("Download complete.")

    def save(self):
        mnist = {}
        for name in filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name in filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(self.path + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist,f)
        print("Save complete.")

    def load(self):
        with open(self.path + "mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


    

# load MNIST data
MNIST_data = h5py.File("../MNISTdata.hdf5", 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
x_test  = np.float32(MNIST_data['x_test'][:])
y_test  = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
MNIST_data.close()


# stack together for next step
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))


# one-hot encoding
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


# number of training set
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]


# shuffle training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]