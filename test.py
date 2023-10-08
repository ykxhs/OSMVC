import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

# MNIST-USPS
# BDGP .
# Caltech-2V
# Caltech-3V .
# Caltech-4V
# Caltech-5V .
# Prokaryotic .
# Cifar10 .
# Cifar100
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--data_path', default="D:/cyy/dataset/MVC_data") # 数据存放位置
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--feature_dim", default=512)
args = parser.parse_args()
if args.dataset == "MNIST-USPS":
    args.mse_epochs = 100
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.feature_dim = 256
    args.mse_epochs = 200
    args.con_epochs = 21
    seed = 5
if args.dataset == "Caltech-2V":
    args.feature_dim = 256
    args.mse_epochs = 250
    args.con_epochs = 80
    seed = 5
if args.dataset == "Caltech-3V":
    args.feature_dim = 256
    args.mse_epochs = 200
    args.con_epochs = 100
    seed = 5
if args.dataset == "Caltech-4V":
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 5
if args.dataset == "Caltech-5V":
    args.temperature_l = 0.8
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 5
if args.dataset == "Prokaryotic":
    args.feature_dim = 256
    args.mse_epochs = 40
    args.con_epochs = 5
    seed = 5
if args.dataset == "Cifar10":
    args.mse_epochs = 100
    args.con_epochs = 20
    seed = 10
if args.dataset == "Cifar100":
    args.mse_epochs = 200
    args.con_epochs = 45
    seed = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = load_data(args.dataset,args.data_path)
model = Network(view, dims, args.feature_dim, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, isprint=True)
