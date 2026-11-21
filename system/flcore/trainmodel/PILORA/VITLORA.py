import torch
import copy
import shutil
import torch.nn as nn
import torch.optim as optim
import math
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from pathlib import Path
from flcore.trainmodel.PILORA.VLT import *
from flcore.trainmodel.PILORA.utils import *
from flcore.trainmodel.PILORA.update import *
from tqdm import tqdm
# from iCIFAR100 import iCIFAR100
from flcore.trainmodel.PILORA.CPN import *
# from t_sneplot import *

class vitlora:
    def __init__(self, args, file_name, model, task_size, device):
        self.data_dir = '../../data'
        self.file_name = file_name
        self.args = args
        self.epochs = args.local_epochs
        self.model = model
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.global_model = VLT(modelname='vit_base_patch16_224_dino',
                                num_classes=self.args.num_classes,
                                pretrained=bool(getattr(args, "pretrained", True)),
                                r = 40,
                                lora_layer = [0],
                                g_model=True,
                                num_lora_tasks=1)
        # self.trans_train = transforms.Compose([
        #                     transforms.Resize(224),
        #                     transforms.RandomHorizontalFlip(p=0.5),
        #                     transforms.ColorJitter(brightness=0.24705882352941178),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        # self.trans_test = transforms.Compose([
        #                     transforms.Resize(224),
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        # self.train_set = iCIFAR100(self.data_dir, train=True, download=True,
        #                         transform=self.trans_train)
        # self.valid_set = iCIFAR100(self.data_dir, train=True, download=True,
        #                         test_transform=self.trans_test)
        # self.test_set = iCIFAR100(self.data_dir, train=False, download=True,
        #                         test_transform=self.trans_test)
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.classes = None
        self.old_model = None
        self.list_of_testloader = list()
        self.W_aq = []
        self.W_av = []
        self.W_bq = []
        self.W_bv = []

    def inference(self, model, test_loader):
        if test_loader is None or len(test_loader.dataset) == 0:
            return 0.0, 0.0
        model.eval()
        test_loss = 0.0
        loss_sum = 0.0
        correct = 0.0
        extracted_features = []
        extracted_label = []
        extracted_centers = []
        
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch[-2], batch[-1]
                data, target = data.to(self.device), target.to(self.device)
                extracted_label.extend(target.cpu().data.numpy())
                predict, feature, dist, centers, _ = model(data, mode='test')

                extracted_features.extend(feature.detach().cpu().numpy().tolist())
                extracted_centers = centers.T.detach().cpu().numpy()

                logits, batch_loss = DCE(target, dist, type='test')
                loss_sum += float(batch_loss.item()) * target.size(0)
                pred = torch.max(logits, 1)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        # t_SNEplot(extracted_features, extracted_label, extracted_centers, self.numclass)
        test_loss = loss_sum / max(1, len(test_loader.dataset))
        acc = 100. * correct / len(test_loader.dataset)
        return acc, test_loss
    
    def save_checkpoint(self, state, is_best):
        checkpoint_dir = Path(self.args.out_folder) / "checkpoints" / self.args.store_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = str(checkpoint_dir / "ckpt.pth.tar")
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

    def map_new_class_index(self, y, order):
        return np.array(list(map(lambda x: order.index(x), y)))

    def setup_data(self, shuffle, seed):
        train_targets = self.train_set.targets
        test_targets = self.test_set.targets
        order = [i for i in range(len(np.unique(train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = range(len(order))
        self.class_order = order
        if seed == 0:
            self.class_order = [i for i in range(len(np.unique(train_targets)))]
        print(100*'#')
        print(self.class_order)

        self.class_mask = build_continual_dataset(self.args, self.class_order)
        print(self.class_mask)

        self.train_set.targets = self.map_new_class_index(train_targets, self.class_order)
        self.test_set.targets = self.map_new_class_index(test_targets, self.class_order)

    def beforeTrain(self, current_task, available_labels_current):
        self.model.eval()
        # if current_task == 0:
        #     self.classes = [0, self.numclass]
        # else:
        #     self.classes = [self.numclass-self.task_size, self.numclass]
        self.model.centers_initial(available_labels_current)
        
        # self.test_set = iCIFAR100(self.data_dir, train=False, download=True,
        #                          test_transform=self.trans_test)
        self.model.switch_lora(current_task)
        self.global_model.switch_lora(current_task)
        # self.test_set.getTestData(self.classes)
        # self.test_loader = DataLoader(self.test_set, batch_size=100,
        #                             shuffle=False, num_workers=8)
        # self.list_of_testloader.append(self.test_loader)

        # self.valid_set.getValidData(self.classes)
        # self.valid_loader = DataLoader(self.valid_set, batch_size=100,
        #                             shuffle=False, num_workers=8)

        self.model.train()
        self.model.to(self.device)
        self.global_model.to(self.device)

    def train(self, current_task, tf_writer=None, logger_file=None):
        bst_acc = -1
        description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"
        local_weights = []
        center_lr = self.args.centers_lr
        encoder_lr = self.args.encoders_lr

        best_acc = 0.0  
        early_stopping_patience = 10  
        current_patience = 0  

        epochs = int(self.args.global_rounds)
        for epoch in tqdm(range(epochs)):
            local_weights = []
            feature_list = [[] for _ in range(self.args.num_classes)]
            m = self.args.num_clients
            if getattr(self.args, "participation_mode", "configured") == "drop_one":
                # One client is absent from both its local update and aggregation.
                drop_rng = getattr(self, "_participation_rng", None)
                if drop_rng is None:
                    self._participation_rng = drop_rng = np.random.default_rng(
                        int(getattr(self.args, "seed", 42)) + 982451653)
                excluded = int(drop_rng.integers(m))
                idxs_users = np.asarray([i for i in range(m) if i != excluded])
                print(f"[one-client blackout] client_id={excluded}", flush=True)
            else:
                idxs_users = np.random.choice(range(m), m, replace=False)
            sample_num = []
            # load dataset and user groups

            for idx in idxs_users:
                # train_dataset, user_groups = get_dataset(self.args, train_dataset=self.train_set, m=m,
                #                                          start=self.classes[0], end=self.classes[1], task_num=self.task_size)

                # map_dict = {}
                # for (key, value), new_key in zip(user_groups.items(), idxs_users):
                #     map_dict[new_key] = value
                # user_groups = map_dict

                # sample_num.append(len(user_groups[idx]))

                train_data = getattr(self, "train_data_by_client", None)
                local_model = LocalUpdate(args=self.args, current_task=current_task,
                                          idx=idx,
                                          train_data=None if train_data is None else train_data[idx])
                w, feature_average = local_model.update_weights(
                    model=copy.deepcopy(self.model), old_model=copy.deepcopy(self.old_model), lr_c=center_lr, lr_e=encoder_lr,
                      current_task=current_task, Waq=self.W_aq, Wav=self.W_av)
                local_weights.append(copy.deepcopy(w))
                sample_num.append(local_model.sample_count)
                for key in range(self.args.num_classes):
                    feature_list[key].append(feature_average.get(key, []))
            if not local_weights or sum(sample_num) <= 0:
                raise RuntimeError("PILORA selected no clients with non-empty task data")
            average_weight = [n / sum(sample_num) for n in sample_num]

            # # update global weights
            # self.model.load_state_dict(self.model)
            self.model.load_state_dict(average_weights(local_weights, self.model, self.classes,
                                                        self.args.niid_type, feature_list, average_weight))
            self.global_model = global_server(self.model, self.global_model, self.W_aq, self.W_av, self.W_bq, self.W_bv, current_task)
            # self.model_ave.load_state_dict(average_weights2(local_weights, self.model))

            valid_acc, valid_loss = self.inference(self.global_model, self.valid_loader)
            test_acc, test_loss = self.inference(self.global_model, self.test_loader)

            # center_lr, encoder_lr = center_lr*0.95, encoder_lr*0.95
            # if epoch < self.args.epochs/2:
            #     center_lr = self.args.centers_lr
            #     encoder_lr = self.args.encoders_lr
            # else:
            center_lr = self.args.centers_lr*0.5*(1 + math.cos(epoch * math.pi / epochs))
            encoder_lr = self.args.encoders_lr*0.5*(1 + math.cos(epoch * math.pi / epochs))

            if tf_writer is not None:
                tf_writer.add_scalar('test_acc', test_acc, epoch)
                tf_writer.add_scalar('test_loss', test_loss, epoch)

            output_log = 'After {} global rounds, Test acc: {}, inference loss: {}'.format(
                epoch + 1, test_acc, test_loss)
            if logger_file is not None:
                logger_file.write(output_log + '\n')
                if self.valid_loader is not None:
                    output_log = 'After {} global rounds, Valid acc: {}, Valid loss: {}'.format(
                        epoch + 1, valid_acc, valid_loss)
                    logger_file.write(output_log + '\n')
                logger_file.flush()

            is_best = test_acc > bst_acc
            bst_acc = max(bst_acc, test_acc)
            #print(description.format(test_acc, test_loss, bst_acc))
                
            self.save_checkpoint(self.model.state_dict(), is_best)
            if self.valid_loader is not None and valid_acc > best_acc:
                best_acc = valid_acc
                current_patience = 0 
            elif self.valid_loader is not None:
                current_patience += 1 
                
                if current_patience >= early_stopping_patience and current_task != 0:
                    print("Early stopping triggered. Training stopped.")
                    break 
            
        print(description.format(test_acc, test_loss, bst_acc))

    def afterTrain(self, current_task):
        path = Path(self.args.out_folder) / "models" / self.file_name
        path.mkdir(parents=True, exist_ok=True)
        self.numclass += self.task_size
        filename = path / ('%d_model.pkl' % (self.numclass - self.task_size))
        torch.save(self.global_model, filename)
        for name, param in self.model.named_parameters():
            if 'linear_a_q_{}'.format(current_task) in name:
                self.W_aq.append(param)
            if 'linear_a_v_{}'.format(current_task) in name:
                self.W_av.append(param)
            if 'linear_b_q_{}'.format(current_task) in name:
                self.W_bq.append(param)
            if 'linear_b_v_{}'.format(current_task) in name:
                self.W_bv.append(param)
        self.old_model = torch.load(filename, map_location=self.device, weights_only=False)
        self.old_model.to(self.device)
        self.old_model.eval()
