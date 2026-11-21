#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from utils.data_utils import *
from flcore.trainmodel.PILORA.CPN import *
from sklearn.metrics.pairwise import cosine_similarity


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        _, image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class ViTResizeDataset(Dataset):
    """Convert repository image tensors to the 224px ImageNet ViT contract."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = torch.as_tensor(image).float()
        if image.ndim != 3:
            raise ValueError(f"PILORA expects a 3D image tensor, got shape {tuple(image.shape)}")
        if image.shape[0] not in (1, 3) and image.shape[-1] in (1, 3):
            image = image.permute(2, 0, 1)
        if image.max() > 1.5:
            image = image / 255.0
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear",
                              align_corners=False).squeeze(0)
        mean = image.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = image.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (image - mean) / std, torch.as_tensor(label).long()


class LocalUpdate(object):
    def __init__(self, args, current_task, idx, train_data=None):
        self.args = args
        self.current_task = current_task
        self.id = idx
        # self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
        #                               batch_size=self.args.local_bs, shuffle=True, num_workers=8)
        dataset = train_data if train_data is not None else self.load_dataset(self.current_task)
        self.sample_count = len(dataset)
        self.trainloader = DataLoader(ViTResizeDataset(dataset), self.args.batch_size,
                                      drop_last=False, shuffle=self.sample_count > 0)

    def load_dataset(self, task):
        if self.args.dataset == 'IMAGENET1k':
            train_data = read_client_data_FCL_imagenet1k(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        elif self.args.dataset == 'CIFAR100':
            train_data = read_client_data_FCL_cifar100(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        elif self.args.dataset == 'CIFAR10':
            train_data = read_client_data_FCL_cifar10(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        
        else:
            raise NotImplementedError(f"PILORA does not support {self.args.dataset}")
        return train_data

    def update_weights(self, model, old_model, lr_c, lr_e, current_task, Waq, Wav):
        model.train()

        pg = [p for p in model.parameters() if p.requires_grad]
        train_name = [name for name, param in model.named_parameters() if param.requires_grad]
        network_params = []
        for name, param in model.named_parameters():
            lr = lr_c 
            if name.startswith('encoder'):
                lr = lr_e 
            elif name.startswith('centers'):
                lr = lr_c 
            
            param_group = {'params': [param], 'lr': lr, 'weight_decay': 0.00001}
            network_params.append(param_group)

        optimizer = torch.optim.Adam(network_params)
        label = []
        all_classes = range(int(self.args.num_classes))
        label_feature_dict = {cls: [] for cls in all_classes}
        for iter in range(self.args.local_epochs):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                label.extend(labels.tolist())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                model.zero_grad()
                pred, feature, dist, centers, feat_layer4 = model(images)
                logits, loss_dce = DCE(labels, dist)

                l1_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)  
                ort_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)  

                for name, param in model.named_parameters():
                    if 'linear_a_q_{}'.format(current_task) in name:
                        l1_loss += torch.norm(param, p=1)
                    if 'linear_a_v_{}'.format(current_task) in name:
                        l1_loss += torch.norm(param, p=1)

                
                if old_model is None:
                    loss = loss_dce + 0.01 * l1_loss
                else:
                    for name, param in model.named_parameters():
                        if 'linear_a_q_{}'.format(current_task) in name:
                            for pre_param in Waq:
                                ort_loss += torch.abs(torch.mm(pre_param, param.T)).sum()
                        if 'linear_a_v_{}'.format(current_task) in name:
                            for pre_param in Wav:
                                ort_loss += torch.abs(torch.mm(pre_param, param.T)).sum()
                    loss = loss_dce + 0.01 * l1_loss + 0.5 * ort_loss 
                
                loss.backward()
                optimizer.step()

                for i, lbl in enumerate(labels):
                    lbl = lbl.item()  
                    label_feature_dict[lbl].append(feature[i].detach().cpu().numpy())
        averages = {}
        for cls, values in label_feature_dict.items():
            if not values:  
                averages[cls] = []  
            else:  
                average_value = sum(values) / len(values)
                averages[cls] = average_value
        return model.state_dict(), averages
