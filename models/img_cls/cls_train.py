import torch
import torch.nn as nn
import numpy as np
from cls_model import ImageClsBackbone
from torch.utils.data import Dataset
import torch.nn.functional as F
from glob import glob
import os
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
import os.path as osp
import tqdm

os.environ['CUDA_VISIBLE_DEVICES']= '0'

"""
# source: https://github.com/open-mmlab/OpenPCDet/blob/1f5b7872b03e9e3d42801872bc59681ef36357b5/pcdet/config.py
"""

import yaml
from easydict import EasyDict
from pathlib import Path

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

# set empty as default and add key & values
cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0


class ImgDataset(Dataset):
    def __init__(self, path_cfg=None, split='train', transform=None):
        super().__init__()
        self.cfg = cfg_from_yaml_file(path_cfg, cfg)
        self.split = split # in ['train', 'test']
        self.dict_split = self.get_split_dict(self.cfg.DATASET.PATH_SPLIT[split])
        self.list_path_label = [] # a list of dic
        cls_label_list = []
        weather_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        for dir_seq in self.cfg.DATASET.DIR.LIST_DIR:
            list_seq = os.listdir(dir_seq)
            for seq in list_seq:
                seq_label_paths = sorted(glob(osp.join(dir_seq, seq, 'info_label', '*.txt')))
                seq_label_paths = list(filter(lambda x: (x.split('/')[-1].split('.')[0] in self.dict_split[seq]), seq_label_paths))
                self.list_path_label.extend(seq_label_paths)
                
                desc_path = osp.join(dir_seq, seq, 'description.txt')
                f = open(desc_path, 'r')
                desc = f.readlines()[0]
                f.close()
                weather = desc.split(',')[-1]
                label = weather_list.index(weather)
                cls_label_list = cls_label_list + [label] * len(seq_label_paths)    
        self.cls_label_list = cls_label_list
        self.transform = transform

    def get_split_dict(self, path_split):
        # ./tools/train_test_splitter
        f = open(path_split, 'r')
        lines = f.readlines()
        f.close
        dict_seq = dict()
        for line in lines:
            seq = line.split(',')[0]
            label = line.split(',')[1].split('.')[0]
            if not (seq in list(dict_seq.keys())):
                dict_seq[seq] = []
            dict_seq[seq].append(label)
        return dict_seq
    
    def get_data_indices(self, path_label):
        f = open(path_label, 'r')
        line = f.readlines()[0]
        f.close()

        seq_id = path_label.split('/')[-3]
        rdr_idx, ldr_idx, camf_idx, _, _ = line.split(',')[0].split('=')[1].split('_')

        return seq_id, rdr_idx, ldr_idx, camf_idx
    
    def __getitem__(self, idx):
        path_label = self.list_path_label[idx]
        seq_id, radar_idx, lidar_idx, camf_idx = self.get_data_indices(path_label)
        path_header = path_label.split('/')[:-2]
        path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
        
        image = cv2.imread(path_cam_front)[:,:1280]
        label = self.cls_label_list[idx]
        
        if self.transform:
            image = self.transform(image)
          
        return image, label, path_cam_front

    def __len__(self):
        return len(self.cls_label_list)

network = ImageClsBackbone()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
path_cfg = './configs/cfg_rl_3df_gate.yml'
trainset = ImgDataset(path_cfg=path_cfg, split='train', transform=transform)
testset = ImgDataset(path_cfg=path_cfg, split='test', transform=transform)
print(trainset.__len__(), testset.__len__())

batch_size = 16
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
weather_list = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'light snow', 'heavy snow']

best_acc = 0
for epoch in range(100):
    running_loss = 0.0
    network.train()
    for i, data in tqdm.tqdm(enumerate(trainloader, 0)):
        image, label = data[0].to(device), data[1].to(device)
        dict_item = dict()
        dict_item['cam_front_img'] = image

        optimizer.zero_grad()
        outputs = network(dict_item)
        loss = criterion(outputs['img_cls_output'], label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    correct = 0
    total = 0
    network.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(testloader):
            image, label, path_cam_front = data
            image, label = data[0].to(device), data[1].to(device)
            dict_item = dict()
            dict_item['cam_front_img'] = image
            outputs = network(dict_item)
            _, predicted = torch.max(outputs['img_cls_output'].data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    acc = correct / total

    if acc > best_acc:
        torch.save(network.state_dict(), 
                   './models/img_cls/best' + str(epoch).zfill(3) + '_' + str(acc*100).split('.')[0] + '.pth')
        print('epoch, best acc: ' + str(epoch).zfill(3) + '_' + str(acc))
        best_acc = acc
    else:
        print('epoch, acc: ' + str(epoch).zfill(3) + '_' + str(acc))

