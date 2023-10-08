from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path+'/BDGP.mat')
        data1 = data['X1'].astype(np.float32)
        data2 = data['X2'].astype(np.float32)
        labels = data['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        return [torch.from_numpy(self.x1), torch.from_numpy(self.x2)], torch.from_numpy(self.y)

class MNIST_USPS(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/MNIST_USPS.mat')
        self.Y = data['Y'].astype(np.int32).reshape(5000,)
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1.reshape(-1,784)
        x2 = self.V2.reshape(-1,784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        if self.view == 2:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2)], torch.from_numpy(self.labels)
        if self.view == 3:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5)], torch.from_numpy(self.labels)
        if self.view == 4:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view4)], torch.from_numpy(self.labels)
        if self.view == 5:
            return [torch.from_numpy(self.view1), torch.from_numpy(self.view2), torch.from_numpy(self.view5), torch.from_numpy(self.view4), torch.from_numpy(self.view3)], torch.from_numpy(self.labels)

class Prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

class Cifar10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y
class Cifar100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '/cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    def full_data(self):
        x1 = self.V1
        x2 = self.V2
        x3 = self.V3
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y

def load_data(dataset,data_path):
    if dataset == "BDGP":
        dataset = BDGP(data_path)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS(data_path)
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Caltech-2V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech(f'{data_path}/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Prokaryotic":
        dataset = Prokaryotic(data_path)
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == "Cifar10":
        dataset = Cifar10(data_path)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = Cifar100(data_path)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
