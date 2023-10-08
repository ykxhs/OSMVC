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
Dataname = 'Cifar10'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--data_path', default="D:/cyy/dataset/MVC_data") # 数据存放位置
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--feature_dim", default=512)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dataset in ["BDGP","Caltech-2V","Caltech-3V","Caltech-4V","Caltech-5V","Prokaryotic"]:
    args.feature_dim = 256
dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_path)
model = Network(view, dims, args.feature_dim, class_num, device)
model = model.to(device)

checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, isprint=True)
