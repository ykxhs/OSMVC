import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

# view-cross fusion
class VCF(nn.Module):
    def __init__(self,in_feature_dim,class_num):
        super(VCF,self).__init__()
        TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=in_feature_dim, nhead=1,dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(TransformerEncoderLayer, num_layers=1)
        self.cluster = nn.Sequential(
            nn.Linear(in_feature_dim,class_num),
            nn.Softmax(dim=1)
        )
    def forward(self,C):
        temp = self.TransformerEncoder(C)
        t = self.cluster(temp)
        return t, temp

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        self.As = nn.ParameterList()
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.As.append(nn.Parameter(torch.Tensor(class_num, feature_dim)).to(device))  # 质心
            torch.nn.init.xavier_normal_(self.As[v].data)

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.VCF = VCF(feature_dim * self.view,class_num)
        self.alpha = 1.0

    def forward(self,X):
        Zs = []
        Cs = []
        Ss = []
        X_hat = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            Zs.append(Z)
            X_hat.append(self.decoders[v](Z))

        Z = torch.cat(Zs, dim=1)
        t, _ = self.VCF(Z)
        P = self.target_distribution(t)
        Qs = []
        for v in range(self.view):
            q = 1.0 / (1.0 + torch.sum(torch.pow(Zs[v].unsqueeze(1) - self.As[v], 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            Qs.append(q)
        return X_hat, Cs, P, Qs, Ss

    def forward_plot(self,X):
        Zs = []
        for v in range(self.view):
            Z = self.encoders[v](X[v])
            Zs.append(Z)
        Z = torch.cat(Zs, dim=1)
        t, fusioned_var = self.VCF(Z)
        P = self.target_distribution(t)
        preds = torch.argmax(P, dim=1)

        return Z, fusioned_var, preds

    def target_distribution(self,p):
        weight = p ** 2 / p.sum(0)
        return (weight.t() / weight.sum(1)).t()

