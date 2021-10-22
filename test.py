import os
import sys
import csv
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

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

if __name__ == '__main__':
    img_size = 256
    device = 'cuda'
    test_trainset = Jungle('oulu_npu_cropped/test', transform = transforms.Compose([
        transforms.Resize((img_size ,img_size )),
        transforms.ToTensor(),
        transforms.Normalize([0.0,],[1.0,])
    ]))

    shared_spoof_path = 'S18'
    c_path = 'C18'

    test_loader = DataLoader(test_trainset, batch_size = 16, shuffle = False)
    shared_spoof = torch.load(shared_spoof_path).to(device)
    C = torch.load(c_path).to(device)

    shared_spoof.eval()
    C.eval()

    
    pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            im = data
            im = im.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)

            result = shared_spoof(im)
            result = C(result).cpu().numpy()

            for i in range(len(result)):
                pred.append(result[i][0])
    
    pred_oulu_path = 'oulu.csv'
    with open(pred_oulu_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        title = ['video_id', 'label']
        writer.writerow(title)
        for i in range(len(pred)):
            if i == 0:
                temp = [pred[i]]

            elif i % 11 == 0 :
                temp = [pred[i]]

            elif i % 11 == 10:
                temp.append(pred[i])

                id = str(int(i/11))
                if len(id) == 1:
                    id = '000' + id
                elif len(id) == 2:
                    id = '00' + id
                else:
                    id = '0' + id
                ans = 0
                for i in temp:
                    ans+=i
                ans/=11
                title = [id, ans]
                writer.writerow(title)
            else:
                temp.append(pred[i])
    print('oulu done')
    
    
    siw_trainset = Tony('siw_test', transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.0,],[1.0,])
    ]))

    siw_loader = DataLoader(siw_trainset, batch_size = 16, shuffle = False)

    pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(siw_loader):
            im = data
            im = im.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)

            result = shared_spoof(im)
            result = C(result).cpu().numpy()

            for i in range(len(result)):
                pred.append(result[i][0])
    
    pred_siw_path = 'siw.csv'
    with open(pred_siw_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        title = ['video_id', 'label']
        writer.writerow(title)
        for i in range(len(pred)):
            if i == 0:
                temp = [pred[i]]

            elif i % 10 == 0 :
                temp = [pred[i]]

            elif i % 10 == 9:
                temp.append(pred[i])

                id = str(int(i/10))
                if len(id) == 1:
                    id = '000' + id
                elif len(id) == 2:
                    id = '00' + id
                else:
                    id = '0' + id
                ans = 0
                for i in temp:
                    ans+=i
                ans/=10
                title = [id, ans]
                writer.writerow(title)
            else:
                temp.append(pred[i])
    print('siw done')

    pred = []
    with torch.no_grad():
        for batch_idx, data in enumerate(siw_loader):
            im = data
            im = im.to(device)
            im = im.expand(im.data.shape[0], 3, 256, 256)

            result = shared_spoof(im)
            result = C(result)
            result = result.max(1, keepdim = True)[1].cpu().numpy()
            for i in range(len(result)):
                pred.append(result[i][0])
    
    pred_bonus_path = 'bonus.csv'
    with open(pred_bonus_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        title = ['video_id', 'label']
        writer.writerow(title)
        count = 0
        for i in range(len(pred)):
            # print(pred[i])
            if i == 0:
                temp = [pred[i]]

            elif i % 10 == 0 :
                temp = [pred[i]]

            elif i % 10 == 9:
                temp.append(pred[i])

                id = str(int(i/10))
                if len(id) == 1:
                    id = '000' + id
                elif len(id) == 2:
                    id = '00' + id
                else:
                    id = '0' + id
                ans = 0

                label_0 = 0
                label_1 = 0
                label_2 = 0
                for i in temp:
                    if i ==0:
                        label_0+=1
                    elif i ==1:
                        label_1+=1
                    else:
                        label_2+=1

                if label_0 >= 1:
                    ans = 0
                elif label_1 > 5:
                    ans = 1
                elif label_2 > 5:
                    ans = 2
                else:
                    if label_1 == 4:
                        ans = 1
                    else:
                        ans=2
                    count+=1
                title = [id, ans]
                writer.writerow(title)
            else:
                temp.append(pred[i])
    print('bonus done')
