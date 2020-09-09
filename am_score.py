import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from tqdm import tqdm
from torchvision.models.inception import inception_v3
from PIL import Image
import torchvision.transforms.functional as TF
import pudb
import numpy as np
import os
from scipy.stats import entropy
import random

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    print("Loading model")
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm(range(splits)):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(py, pyx) + entropy(pyx))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    class Dataset_manual(torch.utils.data.Dataset):
        def __init__(self, path):
            self.files = os.listdir(path)
            self.root_dir = path

        def __getitem__(self, index):
            image = Image.open(self.root_dir+"/"+self.files[index])
            x = TF.to_tensor(image)

            return x

        def __len__(self):
            return len(self.files)

    class Dataset_list(torch.utils.data.Dataset):
        def __init__(self, path, n = 1000):
            f = open(path, "r")
            self.files = f.readlines()
            f.close()
            self.files = [x.strip() for x in self.files]
            random.shuffle(self.files)
            self.files = self.files[:n]


        def __getitem__(self, index):
            image = Image.open("../"+self.files[index])
            x = TF.to_tensor(image)
            return x

        def __len__(self):
            return len(self.files)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )

    # x = IgnoreLabelDataset(cifar)
    txt = "../pic_list/['Smiling', 'Male']~[]"
    # x = Dataset_manual(txt)
    x = Dataset_list(txt)
    print ("Calculating Inception Score...")
    print (inception_score(x, cuda=False, batch_size=32, resize=True, splits=10))
    print(txt.split("/")[-1])
