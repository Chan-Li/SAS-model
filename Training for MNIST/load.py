#Fromhttps://blog.csdn.net/weixin_40522523/article/details/82823812
#The file to load MNIST dataset
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
from struct import unpack
import os

TRAIN_IMAGES = str(os.getcwd())+'/mnist/train-images.idx3-ubyte'
TRAIN_LABELS = str(os.getcwd())+'/mnist/train-labels.idx1-ubyte'
TEST_IMAGES = str(os.getcwd())+'/mnist/t10k-images.idx3-ubyte'
TEST_LABELS = str(os.getcwd())+'/mnist/t10k-labels.idx1-ubyte'


def __read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def __read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab
    
def __normalize_image(image):
    img = image.astype(np.float32) / 255.0
    return img

def __one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab
def load_mnist(train_image_path=TRAIN_IMAGES, train_label_path=TRAIN_LABELS, test_image_path=TEST_IMAGES, test_label_path=TEST_LABELS, normalize=True, one_hot=False):
    image = {
        'train' : __read_image(train_image_path),
        'test'  : __read_image(test_image_path)
    }

    label = {
        'train' : __read_label(train_label_path),
        'test'  : __read_label(test_label_path)
    }
    
    if normalize:
        for key in ('train', 'test'):
            image[key] = __normalize_image(image[key])

    if one_hot:
        for key in ('train', 'test'):
            label[key] = __one_hot_label(label[key])

    return (image['train'], label['train']), (image['test'], label['test'])

class mnistdataset(Dataset):
    def __init__(self,train_data,train_label):
        self.data = train_data
        self.label = train_label
        self.len = len(train_data)
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return self.len

def show_image(array):
    import matplotlib.pyplot as plt
    img = 255*array.reshape(28,28)
    plt.imshow(img,cmap='Greys')
    plt.show()
def get_mnistdataset(one_hot=False):
    """return a tuple of two elements,first is the training set,second is test set"""
    mnist = load_mnist(one_hot=one_hot)
    train_data_ =mnist[0][0]
    train_label_ = mnist[0][1]

    test_data_ = mnist[1][0]
    test_label_ = mnist[1][1]
    mnistdataset_train = mnistdataset(train_data=train_data_,train_label=train_label_)
    mnistdataset_test = mnistdataset(train_data=test_data_,train_label=test_label_)
    return mnistdataset_train,mnistdataset_test
def get_mnistdataloader(one_hot=False,batch_size_=32,shuffle=False):
    """get a mnist dataloader,the first value is  train ,second is test"""
    mnistdataset_train,mnistdataset_test=get_mnistdataset(one_hot=one_hot)
    mnistdataloader_train = DataLoader(mnistdataset_train,batch_size=batch_size_,num_workers=4,shuffle=shuffle)
    mnistdataloader_test = DataLoader(mnistdataset_test,batch_size=batch_size_,num_workers=4,shuffle=shuffle)
    return mnistdataloader_train,mnistdataloader_test
