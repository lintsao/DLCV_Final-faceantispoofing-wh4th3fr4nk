import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import sys
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import argparse
import csv
from sklearn.metrics import roc_auc_score

class DSN_dataset(Dataset): 
    def __init__(self, root, domain, transform = None):
        self.transform = transform
        self.domain = domain
        self.common_path = root       
        self.filename = sorted(os.listdir(self.common_path))
        self.source = [number for number in self.filename if number[2] == '1']  #把label=1的data都當成source
        self.target = [number for number in self.filename if number[2] == '2']
        self.sourcename = []
        self.sourcelabel = []
        for i in range(len(self.source)):
            dir = sorted(os.listdir(os.path.join(self.common_path, self.source[i])))
            self.sourcename += dir
            self.sourcelabel += [self.source[i].split("_")[2] * len(dir)]
        self.targetname = []
        self.targetlabel = []
        for i in range(len(self.target)):
            dir = sorted(os.listdir(os.path.join(self.common_path, self.target[i])))
            self.targetname += dir
            self.targetlabel += [self.target[i].split("_")[2] * len(dir)]
        if domain == 'source':
            self.len = len(self.sourcename)
        else:
            self.len = len(self.targetname)

    def __getitem__(self, index):
        if self.domain == 'source':
            path = os.path.join(self.common_path, str(self.source[int(index/11)]), str(self.sourcename[index]))
            if self.source[int(index/11)][-1] == '1':
                label = int(0)
            elif self.source[int(index/11)][-1] == '2':
                label = int(1)
            elif self.source[int(index/11)][-1] == '3':
                label = int(1)
            elif self.source[int(index/11)][-1] == '4':
                label = int(2)
            else:
                label = int(2)
        else:
            path = os.path.join(self.common_path, str(self.target[int(index/11)]), str(self.targetname[index]))
            if self.target[int(index/11)][-1] == '1':
                label = int(0)
            elif self.target[int(index/11)][-1] == '2':
                label = int(1)
            elif self.target[int(index/11)][-1] == '3':
                label = int(1)
            elif self.target[int(index/11)][-1] == '4':
                label = int(2)
            else:
                label = int(2)        

        im = Image.open(path)                 
        if self.transform is not None:
            im = self.transform(im)

        return im, torch.tensor(label,dtype=torch.long), torch.tensor(int(self.source[int(index/11)].split("_")[2] ),dtype=torch.long)#id label

    def __len__(self):

        return(self.len)

class Jungle(Dataset): 
    def __init__(self, root, transform = None):
        self.transform = transform
        self.common_path = root       
        self.filename = sorted(os.listdir(self.common_path))
        self.name = []
        for i in range(len(self.filename)):
            dir = sorted(os.listdir(os.path.join(self.common_path, self.filename[i])))
            self.name += dir
        self.len = len(self.name)

    def __getitem__(self, index):
        path = os.path.join(self.common_path, str(self.filename[int(index/11)]), str(self.name[index]))

        im = Image.open(path)                 
        if self.transform is not None:
            im = self.transform(im)

        return im

    def __len__(self):

        return(self.len)

class Tony(Dataset): 
    def __init__(self, root, transform = None):
        self.transform = transform
        self.common_path = root       
        self.filename = sorted(os.listdir(self.common_path))
        self.name = []
        for i in range(len(self.filename)):
            dir = sorted(os.listdir(os.path.join(self.common_path, self.filename[i])))
            self.name += dir
        self.len = len(self.name)

    def __getitem__(self, index):
        path = os.path.join(self.common_path, str(self.filename[int(index/10)]), str(self.name[index]))

        im = Image.open(path)                 
        if self.transform is not None:
            im = self.transform(im)

        return im

    def __len__(self):

        return(self.len)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse
class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)
        return simse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class classification(nn.Module):
    def __init__(self):
        super(classification, self).__init__()
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.shared_encoder_pred_class(x)
        return x

class domainclass(nn.Module):
    def __init__(self):
        super(domainclass, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.shared_encoder_pred_domain(x)
        return x

class decoder(nn.Module):
    def __init__(self, code_size=512):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(2000, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

class id_class(nn.Module):
    def __init__(self):
        super(id_class, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 21),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.shared_encoder_pred_domain(x)
        return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

number_folder = '9'
shared_spoof_path = os.path.join('./',number_folder,'shared_spoof')
shared_content_path  = os.path.join('./',number_folder,'shared_content')
c_path = os.path.join('./',number_folder,'C')
pred_oulu_path = os.path.join('./',number_folder,'pred_oulu.csv')
pred_siw_path = os.path.join('./', number_folder,'pred_siw.csv')
pred_bonus_path = os.path.join('./',number_folder,'pred_bonus.csv')
print('目前測試Model名稱 : ', number_folder)
print('Share_spoof 儲存路徑 : ', shared_spoof_path)
print('Share_content 儲存路徑 : ', shared_content_path)
print('Share_classify 儲存路徑 : ', c_path)
print('pred_oulu 儲存路徑 : ', pred_oulu_path)
print('pred_siw 儲存路徑 : ', pred_siw_path)
print('pred_bonus 儲存路徑 : ', pred_bonus_path)

img_size = 256

source_trainset = DSN_dataset('oulu_npu_cropped/train', 'source', transform = transforms.Compose([
    transforms.Resize((img_size ,img_size )), # 隨機將圖片水平翻轉
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize([0.0,],[1.0,])
]))
target_trainset = DSN_dataset('oulu_npu_cropped/train', 'target', transform = transforms.Compose([
    transforms.Resize((img_size ,img_size )), # 隨機將圖片水平翻轉
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize([0.0,],[1.0,])
]))
test_trainset = Jungle('oulu_npu_cropped/test', transform = transforms.Compose([
    transforms.Resize((img_size ,img_size )), # 隨機將圖片水平翻轉
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize([0.0,],[1.0,])
]))

session1_loader = DataLoader(source_trainset, batch_size =4, shuffle = True)
session2_loader = DataLoader(target_trainset, batch_size =4, shuffle = True)

private_source = torchvision.models.resnext50_32x4d(pretrained=True).to(device)
private_target = torchvision.models.resnext50_32x4d(pretrained=True).to(device)
shared_content = torchvision.models.resnext50_32x4d(pretrained=True).to(device)
shared_spoof = torchvision.models.resnext101_32x8d(pretrained=True)
C = classification().to(device)
D = domainclass().to(device)
decode = decoder().to(device)
feature_D = id_class().to(device)
shared_spoof = torch.load('./S18').to(device)
C = torch.load('./C18').to(device)
print('finish model')

"""# TRAIN"""

lr = 0.00001

opt_p_s = optim.Adam(private_source.parameters(), lr=lr)
opt_p_t = optim.Adam(private_target.parameters(), lr=lr)
opt_s_c = optim.Adam(shared_content.parameters(), lr=lr)
opt_s_s = optim.Adam(shared_spoof.parameters(), lr=0.00001)
opt_class = optim.Adam(C.parameters(), lr=0.00001)
opt_domain = optim.Adam(D.parameters(), lr=0.00001)
opt_decode = optim.Adam(decode.parameters(), lr=0.00001)
opt_feature_d = optim.Adam(feature_D.parameters(), lr=0.00001)

class_criterion = nn.CrossEntropyLoss()
label_criterion = nn.BCEWithLogitsLoss()
domain_criterion = nn.BCEWithLogitsLoss()

mse_loss = MSE()
simse_loss = SIMSE()
loss_diff = DiffLoss()
loss_similarity = nn.BCEWithLogitsLoss()

n_epoch =  2
len_dataloader = max(len(session1_loader),len(session2_loader))
best_acc = 0
for epoch in range(n_epoch):  
    private_source.train()
    private_target.train()
    shared_content.train()
    shared_spoof.train()
    feature_D.train()
    C.train()
    D.train()
    decode.train()
    print('session1 -> session2')
    for i, ((source_data, source_label, id_label), (target_data,_,target_label)) in enumerate(zip(session1_loader, session2_loader)):
        source_data = source_data.expand(source_data.data.shape[0], 3, img_size ,img_size )
        target_data = target_data.expand(target_data.data.shape[0], 3, img_size , img_size )
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        id_label = id_label.to(device)
        target_label = target_label.to(device)
        mixed_label =torch.cat([id_label, target_label], dim=0).to(device)
        feature_D.train()
        D.train()

        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0).to(device)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1


        # Step 1 : 訓練 Domain Classifier
        shared_content_feature = shared_content(mixed_data.detach())
        shared_spoof_feature = shared_spoof(mixed_data.detach())
        shared_feature = shared_content_feature+shared_spoof_feature
        domain_logit = D(shared_feature.detach())
        loss = domain_criterion(domain_logit, domain_label)
        loss.backward()
        opt_domain.step()
        D.eval()

        # Step 2 : 訓練 Id Classifier
        predict_id_feature = feature_D(shared_content_feature.detach())
        dis_loss = class_criterion(predict_id_feature, mixed_label)
        dis_loss.backward()
        opt_feature_d.step()
        opt_feature_d.zero_grad()
        feature_D.eval()

        # Step 3 : 訓練 Share Encoder 和 Class Classifier
        shared_source = shared_feature[:source_data.shape[0]]
        shared_target = shared_feature[source_data.shape[0]:]
        pred_class = C(shared_spoof_feature[:source_data.shape[0]])
        domain_logit = D(shared_feature)
        predict_id_feature = feature_D(shared_content(mixed_data))
        predict_id_spoof = feature_D(shared_spoof(source_data))
        id_content_loss = class_criterion(predict_id_feature, mixed_label)
        id_spoof_loss = class_criterion(predict_id_spoof, id_label)#應該是不會用到

        err_domain = domain_criterion(domain_logit, domain_label) 
        err_class = class_criterion(pred_class, source_label)

        # Step 4 : 訓練 Private Source Encoder, Private Target Encoder, Share Encoder, Share Decoder

        shared_content_feature = shared_content(mixed_data)
        shared_spoof_feature = shared_spoof(mixed_data)
        shared_feature = shared_content_feature +shared_spoof_feature
        shared_source = shared_feature[:source_data.shape[0]]
        shared_target = shared_feature[source_data.shape[0]:]

        private_source_feature = private_source(source_data)
        private_target_feature = private_target(target_data)
        source = torch.cat([shared_source, private_source_feature], dim=1).view(-1,2000,1,1)
        target = torch.cat([shared_target, private_target_feature], dim=1).view(-1,2000,1,1)

        recon_source = decode(source)
        recon_target = decode(target)

        err_sim1 = mse_loss(recon_source, source_data)
        err_sim2 = simse_loss(recon_source, source_data)
        err_sim3 = mse_loss(recon_target, target_data)
        err_sim4 = simse_loss(recon_target, target_data)

       
        #shared diff
        mixed_private = torch.cat([private_source_feature, private_target_feature], dim=0)
        err_diff_1 = loss_diff(shared_feature, mixed_private)#private跟原本的越不同越好
        err_diff_2 = loss_diff(shared_spoof_feature, shared_content_feature)#for feature disentanglement
        
        err = err_class + 0.01*err_sim1 + 0.01*err_sim2 + 0.01*err_sim3 + 0.01*err_sim4 + 0.01*err_diff_1 + 0.01*err_diff_2 - 0.1*err_domain + 0.00001*id_content_loss
        if ( i % 250 == 0): 
            print("\r {}/{} err_class:{:.4f},content={:.4f}, spoof={:.4f},err_diff_1 = {:.7f}, err_diff_2 = {:.7f}, id_loss = {:.4f}".format(i, len(session1_loader), err_class.item(), id_content_loss.item() , id_spoof_loss.item(),err_diff_1.item(), err_diff_2.item(), dis_loss.item()))
        err.backward()

        opt_s_c.step()
        opt_s_s.step()
        opt_p_s.step()
        opt_p_t.step()
        opt_decode.step()
        opt_class.step()

        opt_p_s.zero_grad()
        opt_p_t.zero_grad()
        opt_s_c.zero_grad()
        opt_s_s.zero_grad()
        opt_class.zero_grad()
        opt_domain.zero_grad()
        opt_decode.zero_grad()

    print('session2 -> session1')
    for i, ((source_data, source_label, id_label), (target_data,_,target_label)) in enumerate(zip(session2_loader, session1_loader)):
        source_data = source_data.expand(source_data.data.shape[0], 3, img_size ,img_size )
        target_data = target_data.expand(target_data.data.shape[0], 3, img_size , img_size )
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        id_label = id_label.to(device)
        target_label = target_label.to(device)
        mixed_label =torch.cat([id_label, target_label], dim=0).to(device)
        feature_D.train()
        D.train()

        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0).to(device)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1


        # Step 1 : 訓練 Domain Classifier
        shared_content_feature = shared_content(mixed_data.detach())
        shared_spoof_feature = shared_spoof(mixed_data.detach())
        shared_feature = shared_content_feature+shared_spoof_feature
        domain_logit = D(shared_feature.detach())
        loss = domain_criterion(domain_logit, domain_label)
        loss.backward()
        opt_domain.step()
        D.eval()

        # Step 2 : 訓練 Id Classifier
        predict_id_feature = feature_D(shared_content_feature.detach())
        dis_loss = class_criterion(predict_id_feature, mixed_label)
        dis_loss.backward()
        opt_feature_d.step()
        opt_feature_d.zero_grad()
        feature_D.eval()

        # Step 3 : 訓練 Share Encoder 和 Class Classifier
        shared_source = shared_feature[:source_data.shape[0]]
        shared_target = shared_feature[source_data.shape[0]:]
        pred_class = C(shared_spoof_feature[:source_data.shape[0]])
        domain_logit = D(shared_feature)
        predict_id_feature = feature_D(shared_content(mixed_data))
        predict_id_spoof = feature_D(shared_spoof(source_data))
        id_content_loss = class_criterion(predict_id_feature, mixed_label)
        id_spoof_loss = class_criterion(predict_id_spoof, id_label)#應該不會用到

        err_domain = domain_criterion(domain_logit, domain_label) 
        err_class = class_criterion(pred_class, source_label)

        # Step 4 : 訓練 Private Source Encoder, Private Target Encoder, Share Encoder, Share Decoder

        shared_content_feature = shared_content(mixed_data)
        shared_spoof_feature = shared_spoof(mixed_data)
        shared_feature = shared_content_feature +shared_spoof_feature
        shared_source = shared_feature[:source_data.shape[0]]
        shared_target = shared_feature[source_data.shape[0]:]

        private_source_feature = private_source(source_data)
        private_target_feature = private_target(target_data)
        source = torch.cat([shared_source, private_source_feature], dim=1).view(-1,2000,1,1)
        target = torch.cat([shared_target, private_target_feature], dim=1).view(-1,2000,1,1)

        recon_source = decode(source)
        recon_target = decode(target)

        err_sim1 = mse_loss(recon_source, source_data)
        err_sim2 = simse_loss(recon_source, source_data)
        err_sim3 = mse_loss(recon_target, target_data)
        err_sim4 = simse_loss(recon_target, target_data)

       
        #shared diff
        mixed_private = torch.cat([private_source_feature, private_target_feature], dim=0)
        err_diff_1 = loss_diff(shared_feature, mixed_private)#private跟原本的越不同越好
        err_diff_2 = loss_diff(shared_spoof_feature, shared_content_feature)#for feature disentanglement
        

        err = err_class + 0.01*err_sim1 + 0.01*err_sim2 + 0.01*err_sim3 + 0.01*err_sim4 + 0.01*err_diff_1 + 0.01*err_diff_2 - 0.1*err_domain + 0.00001*id_content_loss
        if ( i % 250 == 0): 
            print("\r {}/{} err_class:{:.4f},content={:.4f}, spoof={:.4f},err_diff_1 = {:.7f}, err_diff_2 = {:.7f}, id_loss = {:.4f}".format(i, len(session1_loader), err_class.item(), id_content_loss.item() , id_spoof_loss.item(),err_diff_1.item(), err_diff_2.item(), dis_loss.item()))
        err.backward()

        opt_s_c.step()
        opt_s_s.step()
        opt_p_s.step()
        opt_p_t.step()
        opt_decode.step()
        opt_class.step()

        opt_p_s.zero_grad()
        opt_p_t.zero_grad()
        opt_s_c.zero_grad()
        opt_s_s.zero_grad()
        opt_class.zero_grad()
        opt_domain.zero_grad()
        opt_decode.zero_grad()

torch.save(shared_content, shared_content_path)
torch.save(shared_spoof, shared_spoof_path)
torch.save(C, c_path)