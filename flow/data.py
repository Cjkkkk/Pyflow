class DataLoader:
    def __init__(self, dataset, batchsize, shuffle):
        self.dataset = dataset
        self.batchsize = batchsize
        self.shuffle = shuffle
    
    def __iter__(self):
        # TODO get batchsize data from dataset
        pass
    
    def __next__(self):
        pass
        
    def __len__(self):
        pass
