import csv

import torch
from matplotlib import pyplot as plt

from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# MNIST-USPS
# BDGP
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Prokaryotic
# Cifar10
# Cifar100
Dataname = "Prokaryotic"
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--data_path', default="./data") # 数据存放位置
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=30)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--lamda_P", default=1.0)
parser.add_argument("--lamda_Q", default=1.0)
parser.add_argument("--feature_dim", default=512)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.mse_epochs = 100
    args.con_epochs = 50
    seed = 10
if args.dataset == "BDGP":
    args.lamda_P = 0.001
    args.lamda_Q = 0.01
    args.feature_dim = 256
    args.mse_epochs = 250
    args.con_epochs = 20
    seed = 5
if args.dataset == "Caltech-2V":
    args.learning_rate = 0.0006
    args.feature_dim = 256
    args.mse_epochs = 300
    args.con_epochs = 100
    seed = 5
if args.dataset == "Caltech-3V":
    args.learning_rate = 0.0001
    args.mse_epochs = 300
    args.con_epochs = 300
    seed = 5
if args.dataset == "Caltech-4V":
    args.lamda_P = 1000
    args.lamda_Q = 1000
    args.learning_rate = 0.0001
    args.mse_epochs = 200
    args.con_epochs = 400
    seed = 10
if args.dataset == "Caltech-5V":
    args.lamda_P = 100
    args.lamda_Q = 1000
    args.learning_rate = 0.00006
    args.mse_epochs = 300
    args.con_epochs = 200
    seed = 10
if args.dataset == "Prokaryotic":
    args.lamda_P = 10
    args.lamda_Q = 10
    args.feature_dim = 64
    args.mse_epochs = 100
    args.con_epochs = 100
    seed = 10
if args.dataset == "Cifar10":
    args.lamda_P = 10
    args.lamda_Q = 1
    args.mse_epochs = 200
    args.con_epochs = 100
    seed = 5
if args.dataset == "Cifar100":
    args.lamda_P = 0.01
    args.lamda_Q = 1
    args.mse_epochs = 200
    args.con_epochs = 45
    seed = 10

def setup_seed(seed):
   random.seed(seed)
   os.environ["PYTHONHASHSEED"] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def pretrain(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def contrastive_train(epoch,lamda_P,lamda_Q):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, Cs, P, Qs, Ss = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(lamda_Q * criterion.forward_label(Qs[v], Qs[w]))
            loss_list.append(lamda_P * F.kl_div(torch.log(P),Qs[v]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss/len(data_loader)

if __name__=="__main__":
    if not os.path.exists('./models'):
        os.makedirs('./models')
    T = 1
    for i in range(T):
        setup_seed(seed)
        dataset, dims, view, data_size, class_num = load_data(args.dataset,args.data_path)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        max_res = [0.0, 0]
        print("ROUND:{} DataName:{} view_num:{}".format(i + 1, Dataname, view))
        model = Network(view, dims, args.feature_dim, class_num, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, class_num, args.temperature_l, device).to(device)
        epoch = 1
        while epoch <= args.mse_epochs:
            pretrain(epoch)
            epoch += 1
        ############################################## Convergence analysis
        folder = "result/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        result = open('{}/{}.csv'.format(folder, Dataname), 'w+')
        writer = csv.writer(result)
        writer.writerow(["ACC", "NMI", "PUR", "loss", "epoch"])
        ############################################## init As
        with torch.no_grad():
            full_data, labels = dataset.full_data()
            Zs = []
            for v in range(view):
                full_data[v] = full_data[v].to(device)
            for v in range(view):
                hidden = model.encoders[v](full_data[v])
                cluster_temp = hidden.detach().cpu()
                # 实测不建议使用 sklearn.cluster 会导致实验的无法复现（应该是精度问题）
                cluster_ids_x, cluster_centers = kmeans(X=cluster_temp, num_clusters=class_num, distance='cosine', device=device)
                model.As[v].data = torch.tensor(cluster_centers).to(device)
        ##############################################
        while epoch <= args.mse_epochs + args.con_epochs:
            loss = contrastive_train(epoch,args.lamda_P,args.lamda_Q)
            acc, nmi, pur = valid(model, device, dataset, view, data_size, isprint=False)
            writer.writerow(["{:.4f}".format(acc), "{:.4f}".format(nmi), "{:.4f}".format(pur), "{:.4f}".format(loss), epoch - args.mse_epochs])
            if acc > max_res[0]:
                max_res = [acc, epoch - args.mse_epochs]
                state = model.state_dict()
                torch.save(state, './models/' + args.dataset + '.pth')
            if epoch == args.mse_epochs + args.con_epochs:
                print('--------args----------')
                for k in list(vars(args).keys()):
                    print('%s: %s' % (k, vars(args)[k]))
                print('--------args----------')
                checkpoint = torch.load('./models/' + args.dataset + '.pth')
                model.load_state_dict(checkpoint)
                valid(model, device, dataset, view, data_size, isprint=True)
            epoch += 1
