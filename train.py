from pickle import FALSE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import false
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
import model

USE_CUDA = torch.cuda.is_available()

def cal_acc(y_true,X_test,net):
  test_dataset = TensorDataset(X_test,torch.tensor(y_true))
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  with torch.no_grad():
    all_pred = []
    all_score = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        if USE_CUDA:
            batch_x = batch_x.cuda()
        output, reconstructions, masked = net(batch_x)
        pred  = np.argmax(masked.cpu().detach().numpy(),axis=1).tolist()
        all_pred += pred
    tn, fp, fn, tp = confusion_matrix(y_test, all_pred).ravel()
    perftab = {"CM": confusion_matrix(y_test, all_pred),
            'ACC': (tp + tn) / (tp + fp + fn + tn),
            'SEN': tp / (tp + fn),
            'PREC': tp / (tp + fp),
            "SPEC": tn / (tn + fp),
            "MCC": matthews_corrcoef(y_test, all_pred),
            "F1": f1_score(y_test, all_pred)
    }
    acc=perftab['ACC']
    recall=perftab['SEN']
    perc=perftab['PREC']
    return acc,recall,perc,perftab

capsule_net = capsule_net_final.CapsNet()
if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters(),lr=0.01,betas=(0.9,0.999))


batch_size = 64
n_epochs = 30
res = 64
fig = pd.read_csv("/data/FCGR/train_fcgr.txt")

str_list=[]
for i in fig['figure']:
  temp=np.array(i.split(" "),dtype=np.float32).reshape(res,res)
  str_list.append(temp)
y=fig['label']

fig_test = pd.read_csv("test_res64.txt")
y_test=fig_test['label']
str_list_test=[]
for i in fig_test['figure']:
  temp=np.array(i.split(" "),dtype=np.float32).reshape(res,res)
  str_list_test.append(temp)

X=torch.tensor(str_list)
X_test=torch.tensor(str_list_test)
X=X.unsqueeze(1)
X_test=X_test.unsqueeze(1)


from torch.utils.data import DataLoader, TensorDataset
dataset=TensorDataset(X,torch.tensor(y))
test_dataset=TensorDataset(X_test,torch.tensor(y_test))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    correct_train = 0
    TP_train, FN_train, FP_train = 0, 0, 0

    for batch_id, (data, target) in enumerate(train_loader):
        #target = torch.eye(2, dtype=torch.float32).index_select(dim=0, index=target.long())

        target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())

        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

        # Calculate the number of correct predictions
        pred_labels = np.argmax(masked.data.cpu().numpy(), 1)
        true_labels = np.argmax(target.data.cpu().numpy(), 1)
        correct_train += np.sum(pred_labels == true_labels)

        # Calculate TP, FN, FP for recall calculation
        TP_train += np.sum((pred_labels == 1) & (true_labels == 1))
        FN_train += np.sum((pred_labels == 0) & (true_labels == 1))
        FP_train += np.sum((pred_labels == 1) & (true_labels == 0))

    # Calculate average accuracy, loss, and recall for the epoch
    avg_train_accuracy = correct_train / len(train_loader.dataset)
    avg_train_loss = train_loss / len(train_loader)
    recall_train = TP_train / (TP_train + FN_train)
    precision_train=TP_train/(TP_train+FP_train)

    print(f"Epoch {epoch + 1}/{n_epochs}, Train Accuracy: {avg_train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}, Train Recall: {recall_train:.4f},Train Precision :{precision_train:.4f}")

    capsule_net.eval()
    with torch.inference_mode():
        avg_test_accuracy,recall_test,precision_test,tab=cal_acc(y_test,X_test,capsule_net)
        print(f"Test Accuracy: {avg_test_accuracy:.4f}, Test Recall: {recall_test:.4f},Test Precision:{precision_test:.4f}")
        torch.save(capsule_net.state_dict(), 'model/capsule_net.pth')
