import time
import torch
import copy
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import *
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from flcore.trainmodel.PILORA.VITLORA import vitlora

from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import numpy as np

import statistics
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from flcore.trainmodel.PILORA.update import ViTResizeDataset


class PILORA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.args = args
        self.args.store_name = '_'.join(
            [args.dataset, 'alpha6', 'lr-' + str(args.centers_lr)])

        self.task_size = args.cpt
        self.cur_path = str(Path(args.out_folder).resolve())
        self.prepare_folders()
        self.file_name = args.store_name

        self.model = copy.deepcopy(args.model)
        self.global_model = vitlora(args, self.file_name, self.model, self.task_size, args.device)

        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):

        # if self.args.num_tasks % self.N_TASKS != 0:
        #     raise ValueError("Set num_task again")

        for task in range(self.args.num_tasks):

            print(f"\n================ Current Task: {task} =================")
            train_sets, test_sets = [], []
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                # print("ahihi " + str(len(available_labels_current)))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

                for i, client in enumerate(self.clients):
                    train_data, label_info = self._task_data(i, task, train=True, labels=True)
                    test_data = self._task_data(i, task, train=False)
                    client.train_data = train_data
                    client.train_targets = [label for _, label in train_data]
                    client.classes_so_far = list(label_info['labels'])
                    client.current_labels = list(label_info['labels'])
                    client.task_dict = {0: list(label_info['labels'])}
                    train_sets.append(train_data)
                    test_sets.append(test_data)
                available_labels = set()
                available_labels_current = set()
                for client in self.clients:
                    available_labels.update(client.classes_so_far)
                    available_labels_current.update(client.current_labels)
                for client in self.clients:
                    client.available_labels = sorted(available_labels)
                    client.available_labels_current = sorted(available_labels_current)
                    client.available_labels_past = []

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    train_data, label_info = self._task_data(i, task, train=True, labels=True)
                    test_data = self._task_data(i, task, train=False)

                    # update dataset
                    self.clients[i].next_task(train_data, label_info) # assign dataloader for new data
                    train_sets.append(train_data)
                    test_sets.append(test_data)
                    # print(self.clients[i].task_dict)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            self.global_model.train_data_by_client = train_sets
            self.global_model.classes = sorted(available_labels_current)
            test_parts = [ViTResizeDataset(ds) for ds in test_sets if len(ds)]
            test_set = ConcatDataset(test_parts) if test_parts else None
            self.global_model.test_loader = (DataLoader(test_set, batch_size=self.args.batch_size,
                                                        shuffle=False, num_workers=0)
                                             if test_set is not None else None)
            # No held-out validation partition exists in the repository's CIFAR100 arrays.
            # Do not reuse test examples for early stopping.
            self.global_model.valid_loader = None

            # ============ train ==============

            # for i in range(self.global_rounds):

            # if task == 0:
            #     old_class = 0
            # else:
            #     old_class = len(class_set[:args.fg_nc + (task - 1) * task_size])

            filename = 'log_task{}.txt'.format(task)
            logger_file = open(os.path.join(self.cur_path + '/logs-VIT-LoRA', self.args.store_name, filename), 'w')
            tf_writer = SummaryWriter(log_dir=os.path.join(self.cur_path + '/logs-VIT-LoRA', self.args.store_name))

            self.global_model.beforeTrain(task, available_labels_current)
            self.global_model.train(task, tf_writer=tf_writer, logger_file=logger_file)
            self.global_model.afterTrain(task)
            logger_file.close()
            tf_writer.close()


                # if i%self.eval_gap == 0:
                #     self.eval(task=task, glob_iter=glob_iter, flag="local")

                # self.Budget.append(time.time() - s_t)
                # print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # Comment for boosting speed for rebuttal run
            
            # if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
            #     if self.args.offlog == True and not self.args.debug:  
            #         self.eval_task(task=task, glob_iter=glob_iter, flag="local")

            #         # need eval before data update
            #         self.send_models()
            #         self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def prepare_folders(self):
        folders_util = [
            os.path.join(self.cur_path, 'logs-VIT-LoRA', self.args.store_name),
            os.path.join(self.cur_path, 'checkpoints', self.args.store_name),
            os.path.join(self.cur_path, 'models', self.args.store_name)]
        for folder in folders_util:
            if not os.path.exists(folder):
                print('creating folder ' + folder)
                os.makedirs(folder, exist_ok=True)

    def _task_data(self, client_id, task, train, labels=False):
        reader = {
            'IMAGENET1k': read_client_data_FCL_imagenet1k,
            'CIFAR100': read_client_data_FCL_cifar100,
            'CIFAR10': read_client_data_FCL_cifar10,
        }.get(self.args.dataset)
        if reader is None:
            raise NotImplementedError(f"PILORA does not support {self.args.dataset}")
        return reader(client_id, task=task, classes_per_task=self.args.cpt,
                      count_labels=labels, train=train)
    
