from flow.data import DataLoader, MNIST
import numpy as np
from PIL import Image

train_loader = DataLoader(
        MNIST('./data', train=True, download=True),
        batch_size=64, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
    np_img = (data[0, 0, :].data * 255).astype(np.uint8)
    img = Image.fromarray(np_img)
    img.save("./result/" + str(batch_idx)+'.png')