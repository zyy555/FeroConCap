import numpy as np

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
USE_CUDA = torch.cuda.is_available()

#conv layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1 #步长
                             )

    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=16, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=2, padding=0
                      )
            for _ in range(num_capsules)])
        # creating 8 conv layers, each perform a separate conv s

    def forward(self, x):
        ls = []
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        #print("usize",u.size())
        #print("usize1,2,0",u.size(1),u.size(2),u.size(0))
        u = u.view(x.size(0), u.size(1)*u.size(3)*u.size(4) , -1)
        #u.view(x.size(0)=batchsize, 8*24*24, -1)
        #print("usize",u.size())
        pri=u.detach().cpu()
        ls.append(pri)
        return self.squash(u)


    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=8*24*24, in_channels=16, out_channels=32):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        #print(in_channels)
        #print(self.in_channels)
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))


    def forward(self, x):
        batch_size = x.size(0)
        #print("x.size",x.size)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        #print("x size",x.size())
        #print(self.W.shape)
        W = torch.cat([self.W] * batch_size, dim=0)
        #print("w size",W.size())
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        v_final = v_j.squeeze(1)
        #print("v_final.shape",v_final.shape)
        v_c = torch.sqrt((v_final ** 2).sum(dim=2, keepdim=True))
        # v_j = v_j.unsqueeze(-1)
        #print("v_j.shape", v_j.shape)
        #print("v_c.shape", v_c.shape)
        return v_final

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((0.5 + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(
            nn.Linear(32 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1*64*64),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes,dim=1)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(2))
        #print("mask", masked.shape)
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        #print("mask",masked.shape)

        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 64, 64)

        return reconstructions, masked

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        margin = self.margin_loss(x, target)
        recon = self.reconstruction_loss(data, reconstructions)
        features = x.squeeze(-1).mean(dim=1)  # shape: (B, out_dim)
        supcon = self.contrastive_loss(features, target)
        # print(f"Margin Loss: {margin.item()}, Reconstruction Loss: {recon.item()}, Contrastive Loss: {supcon.item()}")
        return margin + recon + supcon
        # return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def contrastive_loss(self, x, labels, tao=0.07):
        batch_size = x.size(0)
        device = x.device

        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)

        features = F.normalize(x, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        logits_mask = torch.ones_like(similarity_matrix) - torch.eye(batch_size, device=device)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask * logits_mask
        logits = similarity_matrix / tao
        logits = logits * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1))
        loss = -mean_log_prob_pos.mean()
        return loss * 0.005

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        #with torch.no_grad():
        #    print(f"v_c min: {v_c.min().item()}, max: {v_c.max().item()}, mean: {v_c.mean().item()}")

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        # print(labels)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005