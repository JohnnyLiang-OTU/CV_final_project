import matplotlib.pyplot as plt
import numpy as np

def imshowdef(img):
    img = img + 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()