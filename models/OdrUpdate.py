
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y 
import pdb
import torch.nn.functional as F
from utils.train_utils import get_model
import os
import time
from torch.autograd import Variable
import json
from sklearn.metrics import roc_auc_score
from models.test import test_img_local_all, test_img_local


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalTrain(object):
    def __init__(self, args, logger, dataset_train, dataset_test, dict_users_train, dict_users_test, net):
        self.global_epoch = 0
        self.pickle_record = {"test": {}, "train": {}}
        self.args = args
        self.global_model = net 
        self.global_optim = torch.optim.SGD(self.global_model.parameters(), lr=self.args.lr_global, momentum=self.args.momentum)
        self.global_sched = torch.optim.lr_scheduler.StepLR(self.global_optim, step_size=50, gamma=0.5)
        self.local_models = {}
        self.local_optims = {}
        self.local_scheds = {}
        for idx in range(self.args.num_users):
            self.local_models[idx] = copy.deepcopy(self.global_model)
            self.local_optims[idx] = torch.optim.SGD(self.local_models[idx].parameters(), lr=self.args.lr_local, momentum=self.args.momentum)    
            self.local_scheds[idx] = torch.optim.lr_scheduler.StepLR(self.global_optim, step_size=50, gamma=0.5)
        if self.args.dataset == "eicu" or self.args.dataset == "adult":
            self.nll_loss_func = nn.BCELoss(size_average = False, reduce = False, reduction = None)
            self.log_softmax_func = nn.Sigmoid()
            self.log_softmax_client_func = nn.LogSoftmax(dim=1)
            self.nll_client_loss_func = nn.NLLLoss(size_average = False, reduce = False, reduction = None)
        else:
            self.nll_loss_func = nn.NLLLoss(size_average = False, reduce = False, reduction = None)
            self.log_softmax_func = nn.LogSoftmax(dim=1)
            self.log_softmax_client_func = nn.LogSoftmax(dim=1)
            self.nll_client_loss_func = nn.NLLLoss(size_average = False, reduce = False, reduction = None)
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users_train = dict_users_train
        self.dict_users_test = dict_users_test
        self.set_data(dataset_train, dataset_test, dict_users_train, dict_users_test)
        self.logger = logger
        self.all_args_save(args)
    def set_data(self, dataset_train, dataset_test, dict_users_train, dict_users_test):
        if 'femnist' in self.args.dataset or 'sent140' in self.args.dataset:
            self.train_keys = list(dataset_train.keys())
            self.test_keys =  list(dataset_test.keys())
            self.dataloaders_train = [enumerate(DataLoader(DatasetSplit(dataset_train[i], np.ones(len(dataset_train[i]['x'])), name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True) ) for i in self.train_keys]
            self.dataloaders_test = [enumerate(DataLoader(DatasetSplit(dataset_test[i], np.ones(len(dataset_test[i]['x'])), name=self.args.dataset), batch_size=len(dataset_test[i]['x']), shuffle=True) ) for i in self.test_keys]

        elif self.args.dataset == 'adult':
            self.dataloaders_train = [enumerate(DataLoader(DatasetSplit(dataset_train, dict_users_train[idxs]), batch_size = len(dict_users_train[idxs]), shuffle=True)) for idxs in range(self.args.num_users)]
            self.dataloaders_test = [enumerate(DataLoader(DatasetSplit(dataset_test, dict_users_test[idxs]), batch_size=len(dict_users_test[idxs]), shuffle=False, drop_last=False)) for idxs in range(self.args.num_users)]

        elif self.args.dataset == 'eicu':
            self.dataloaders_train = [enumerate(DataLoader(DatasetSplit(dataset_train, dict_users_train[idxs]), batch_size = int(len(dict_users_train[idxs])), shuffle=True)) for idxs in range(self.args.num_users)]
            self.dataloaders_test = [enumerate(DataLoader(DatasetSplit(dataset_test, dict_users_test[idxs]), batch_size=len(dict_users_test[idxs]), shuffle=False, drop_last=False)) for idxs in range(self.args.num_users)]

        else:
            self.dataloaders_train = [enumerate(DataLoader(DatasetSplit(dataset_train, idxs), batch_size=self.args.local_bs, shuffle=True) ) for key, idxs in dict_users_train.items()]
            self.dataloaders_test = [enumerate(DataLoader(DatasetSplit(dataset_test, idxs), batch_size=len(idxs), shuffle=True) ) for key, idxs in dict_users_test.items()]


    def all_args_save(self, args):
        with open(os.path.join(self.args.target_dir, "args.json"), "w") as f:
            args_dict = copy.deepcopy(vars(args))
            del args_dict["device"]
            json.dump(args_dict, f, indent = 2)


    def get_input(self, user, train = True):
        if train:
            try: 
                _, (data, label) = self.dataloaders_train[user].__next__()
            except:                
                if 'femnist' in self.args.dataset or 'sent140' in self.args.dataset:
                    self.dataloaders_train[user] = enumerate(DataLoader(DatasetSplit(self.dataset_train[self.train_keys[user]], np.ones(len(self.dataset_train[self.train_keys[user]]['x'])), name = self.args.dataset), batch_size=self.args.local_bs, shuffle=True ))
                elif self.args.dataset == 'adult':
                    self.dataloaders_train[user] = enumerate(DataLoader(DatasetSplit(self.dataset_train, self.dict_users_train[user]), batch_size = len(self.dict_users_train[user]), shuffle=False))
            
                elif  self.args.dataset == 'eicu':
                    self.dataloaders_train[user] = enumerate(DataLoader(DatasetSplit(self.dataset_train, self.dict_users_train[user]), batch_size = int(len(self.dict_users_train[user])), shuffle=True))
            
                else:
                    self.dataloaders_train[user] = enumerate(DataLoader(DatasetSplit(self.dataset_train, self.dict_users_train[user]), batch_size=self.args.local_bs, shuffle=True) )
                _, (data, label) = self.dataloaders_train[user].__next__()
        else:
            try:
                _, (data, label) = self.dataloaders_test[user].__next__()
            except:
                if 'femnist' in self.args.dataset or 'sent140' in self.args.dataset:
                    self.dataloaders_test[user] = enumerate(DataLoader(DatasetSplit(self.dataset_test[self.test_keys[user]], np.ones(len(self.dataset_test[self.test_keys[user]]['x'])), name = self.args.dataset), batch_size=len(self.dataset_test[self.test_keys[user]]['x']), shuffle=True ))
                
                elif self.args.dataset == 'adult' or self.args.dataset == 'eicu':
                    self.dataloaders_test[user] = enumerate(DataLoader(DatasetSplit(self.dataset_test, self.dict_users_test[user]), batch_size=len(self.dict_users_test[user]), shuffle=False, drop_last=False))
                
                else:
                    self.dataloaders_test[user] = enumerate(DataLoader(DatasetSplit(self.dataset_test, self.dict_users_test[user]), batch_size=len(self.dict_users_test[user]), shuffle=False))  
                _, (data, label) = self.dataloaders_test[user].__next__()
        
        return data, label


    def model_save(self, model_type, model_name = None):
        if model_name == None:
            model_name = str(self.global_epoch)
        if model_type == "global":
            states = {'epoch':self.global_epoch,
                  'model':self.global_model.state_dict(),
                  'optim':self.global_optim.state_dict()}
            os.makedirs(self.args.global_model_dir, exist_ok = True)
            filepath = os.path.join(self.args.global_model_dir, model_name) 
        else:
            states = {str(user): {'model': self.local_model[user], 'optim': self.local_optims[user]} for user in self.args.num_users}
            states["epoch"] = self.global_epoch
            os.makedirs(self.args.local_model_dir, exist_ok = True)
            filepath = os.path.join(self.args.local_model_dir, model_name)
        
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> {} model are saved in '{}' (epoch {})".format(model_type, filepath, self.global_epoch))    


    def model_load(self, model_type, model_dir):
        with open(model_dir, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.args.device)

        if model_type == "global":
            self.global_model.load_state_dict(checkpoint['model'])
            self.global_optim.load_state_dict(checkpoint['optim'])
            self.global_epoch = checkpoint["epoch"]
            self.logger.info("=> {} model load, checkpoint found at {}".format(model_type, model_dir))
        elif model_type == "local":
            self.local_models = {}
            self.local_optims = {}
            for idx in range(self.args.num_users):
                self.local_models[idx] = copy.deepcopy(self.global_model)
                self.local_models[idx].load_state_dict(checkpoint['model'])
                self.local_optims[idx] = torch.optim.SGD(self.local_models[idx].parameters(), lr=self.args.lr_local, momentum=0.5)
            self.logger.info("=> {} model load, checkpoint found at {}".format(model_type, model_dir))
        else:
            self.logger.info("error model_type: {}.".format(model_type))


    def global_train(self):
        self.global_model.train()
        for epoch in range(self.args.global_epochs):

            global_images = []
            global_labels = []
            global_clients = []

            if self.args.dataset == 'eicu' or self.args.dataset == 'adult':
                sele_users =[ user for user in range(self.args.num_users)]
            elif self.args.num_users >=50:
                sele_users = np.random.choice(range(self.args.num_users), max(int(self.args.frac * self.args.num_users), 1), replace=False)
            
            for idx in sele_users:
                images, labels = self.get_input( idx, train = True)
                
                if self.args.dataset == "adult" or self.args.dataset == "eicu":
                    images, labels = images.float().to(self.args.device), labels.float().to(self.args.device)
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.odr:
                    w, adv_images, adv_labels = self.gene_distribution(self.global_model, self.global_optim, images, labels)
                    global_images.append(adv_images)
                    global_labels.append(adv_labels)
                else:
                    global_images.append(images)
                    global_labels.append(labels)                    
                global_clients.append((torch.zeros_like(labels) + idx).to(torch.int64))
            
            outputs = [ self.global_model(images) for images in global_images ]
            log_probs = [ output[0] for output in outputs ]
            # global_losses_label = [ torch.mean(self.nll_loss_func(self.log_softmax_func(log_probs[idx]), global_labels[idx])) for idx in range(len(global_images)) ]
            
            global_losses_label = [F.log_softmax(log_prob, dim = 1) for log_prob in log_probs]
            global_losses_label = [nn.CrossEntropyLoss()(global_losses_label[idx], global_labels[idx]) for idx in range(len(global_images))]

            if self.args.dataset == "adult":
                loss_client = torch.tensor(0)
            else:
                log_clients = [ output[1] for output in outputs ]
                global_losses_client = [ torch.mean(self.nll_client_loss_func(self.log_softmax_client_func(log_clients[idx]), global_clients[idx])) for idx in range(len(global_images)) ]
                loss_client = torch.mean(torch.stack(global_losses_client))
            global_accs = [self.acc_auc(log_probs[idx], global_labels[idx])[0] for idx in range(len(global_images)) ]
            global_aucs = [self.acc_auc(log_probs[idx], global_labels[idx])[1] for idx in range(len(global_images)) ]
            self.global_optim.zero_grad()
            loss_label = torch.mean(torch.stack(global_losses_label))
            loss = loss_label + self.args.GRL * loss_client


            loss.backward()
            self.global_optim.step()
            # self.global_sched.step()
            self.logger.info("global train: epoch: {}, label_loss: {}, client_loss: {}, acc: {}, auc: {}".format(epoch, loss_label.item(), loss_client.item(), np.mean(global_accs), np.mean(global_aucs)))
            self.global_epoch+=1
            if self.global_epoch % self.args.eval_epoch == 0:
                self.test(test_global_model = True)

            if self.global_epoch >= 1500:
                self.args.odr = True

    def local_train(self):
        if self.args.train_mode == "global":
            self.local_models[self.args.target_user] = copy.deepcopy(self.global_model)
            self.local_optims[self.args.target_user] = torch.optim.SGD(self.local_models[self.args.target_user].parameters(), lr=self.args.lr_local, momentum=0.5)  
        for epoch in range(self.args.local_epochs):
            images, labels = self.get_input( self.args.target_user, train = True)
            if self.args.dataset == "adult" or self.args.dataset == "eicu":
                images, labels = images.float().to(self.args.device), labels.float().to(self.args.device)
                select_lambdas = torch.tensor(np.random.uniform(0,1,int(len(images)/2)*2)).float().to(self.args.device)
            else:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                select_lambdas = torch.tensor(np.random.uniform(0,1,int(len(images)/2)*2)).float().to(self.args.device)
            


            # ############### mix up
            # if self.args.mix_up: 
            #     if len(images.shape) == 2:
            #         img_lambs = select_lambdas.view(-1, 1).repeat(1, images.shape[1])
            #     elif len(images.shape) == 4:
            #         img_lambs = select_lambdas.view(-1, 1,1,1).repeat(1, images.shape[1], images.shape[2], images.shape[3])
                
            #     images_0 = img_lambs[: int(len(images)/2)] * images[: int(len(images)/2)] + (1-img_lambs[: int(len(images)/2)]) * images[int(len(images)/2): int(len(images)/2)*2]
            #     labels_0 = select_lambdas[: int(len(images)/2)] * labels[: int(len(labels)/2)] + (1-select_lambdas[: int(len(images)/2)]) * labels[int(len(labels)/2): int(len(labels)/2)*2]
            #     images_1 = img_lambs[ int(len(images)/2):] * images[: int(len(images)/2)] + (1-img_lambs[int(len(images)/2): ]) * images[int(len(images)/2): int(len(images)/2)*2]
            #     labels_1 = select_lambdas[int(len(images)/2): ] * labels[: int(len(labels)/2)] + (1-select_lambdas[int(len(images)/2): ]) * labels[int(len(labels)/2): int(len(labels)/2)*2]
            #     images = torch.cat([images_0, images_1], dim = 0)
            #     if self.args.dataset == "adult":
            #         labels = torch.cat([labels_0, labels_1], dim = 0).to(torch.float32)
            #     else:
            #         labels = torch.cat([labels_0, labels_1], dim = 0).to(torch.long)
            # else:
            #     pass

            log_probs = self.local_models[self.args.target_user](images)[0]
            loss_list = self.nll_loss_func(self.log_softmax_func(log_probs), labels)
            
            loss_user = torch.mean(loss_list)
            acc, auc = self.acc_auc(log_probs, labels)

            if self.args.odr:
                w, adv_images, adv_labels = self.gene_distribution(self.global_model, self.global_optim, images, labels)
                adv_log_probs = self.local_models[self.args.target_user](adv_images)[0]
                adv_loss = torch.mean(self.nll_loss_func(self.log_softmax_func(adv_log_probs), adv_labels))

            else:
                adv_loss = 0

            # self.logger.info("local train: epoch: {}, loss: {}, acc: {}, auc: {}".format(epoch, loss_user.item(), acc, auc))
            self.local_optims[self.args.target_user].zero_grad()
            (loss_user + self.args.alpha_odr * adv_loss) .backward()
            self.local_optims[self.args.target_user].step()
            self.global_epoch+=1
            if self.global_epoch % self.args.eval_epoch == 0:
                self.test(test_global_model = False, test_user_set = [self.args.target_user])


    def gene_distribution(self, model, optim, data, labels, w = None):
        if self.args.perturb_w:
            if w == None:
                w = torch.ones(data.size(0)).to(self.args.device)
                w.requires_grad_()
            else:
                w.requires_grad_()
            log_probs = model(data)[0]
            loss_list = self.nll_loss_func(self.log_softmax_func(log_probs), labels)
            
            # idx = loss_list.topk(400)[1]
            # loss = torch.mean(w * loss_list) - self.args.lambda_odr * torch.mean((w-1)**2 * torch.mean(data **2))
            loss = torch.mean(w * loss_list) - self.args.lambda_odr * torch.mean((w-1)**2)
            ### optimize w
            loss.backward()
            w = w + self.args.lr_odr_w * w.grad.data 
            w = w/w.max()
            w.data = torch.clamp(w.data, 0.0, 1.0)
            return w.data, data, labels
        else:
            w = torch.ones(data.size(0)).to(self.args.device)
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).detach().to(self.args.device)
            x_adv.requires_grad_()
            for k in range(self.args.attack_steps):
                loss_adv = torch.mean(self.nll_loss_func(self.log_softmax_func(model(x_adv)[0]), labels ))
                loss_adv.backward()
                eta = 0.01 * x_adv.grad.sign()
                x_adv.data = x_adv.data + eta.data
                x_adv.data = torch.min(torch.max(x_adv.data, data - self.args.eps), data + self.args.eps)
                x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
                x_adv.grad.zero_()
            return w.data, x_adv.data, labels 
 

    def acc_auc(self, prob, Y):

        if self.args.dataset == "adult" or self.args.dataset == "eicu":
            y_pred = self.log_softmax_func(prob).data >=0.5
            users_acc = torch.mean((y_pred==Y).float()).item()
        else:
            y_pred = prob.data.max(1)[1]
            users_acc = torch.mean((y_pred==Y).float()).item()
        if self.args.dataset == "eicu":
            prob = self.log_softmax_func(prob)
            users_auc = roc_auc_score(Y.data.cpu().numpy(), prob.data.cpu().numpy())
        else:
            users_auc = 0
        return users_acc, users_auc



    def odr_train(self):
        for epoch in range(self.args.global_epochs):
            client_losses = {}
            local_losses = []
            local_adver_losses = []
            local_accs = []
            local_aucs = []
            local_ws = []
            local_images = []
            local_labels = []
            local_adver_images = []
            local_adver_labels = []

            if self.args.dataset == 'eicu' or self.args.dataset == 'adult':
                sele_users =[ user for user in range(self.args.num_users)]
            elif self.args.num_users >=50:
                sele_users = np.random.choice(range(self.args.num_users), max(int(self.args.frac * self.args.num_users), 1), replace=False)
            
            for idx in sele_users:
                images, labels = self.get_input( idx, train = True)
                if self.args.dataset == "adult" or self.args.dataset == "eicu":
                    images, labels = images.float().to(self.args.device), labels.float().to(self.args.device)
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                log_probs = self.local_models[idx](images)[0]
                loss_list = self.nll_loss_func(self.log_softmax_func(log_probs), labels)
                acc, auc = self.acc_auc(log_probs, labels)
                local_accs.append(acc)
                local_aucs.append(auc)
                local_losses.append(torch.mean(loss_list))
                local_images.append(images)
                local_labels.append(labels)

                if self.args.odr:
                    w, adv_images, adv_labels = self.gene_distribution(self.global_model, self.global_optim, images, labels)
                    adv_log_probs = self.local_models[idx](adv_images)[0]
                    adv_loss = torch.mean(self.nll_loss_func(self.log_softmax_func(adv_log_probs), adv_labels))
                    local_adver_losses.append(adv_loss)
                    local_adver_images.append(adv_images)
                    local_adver_labels.append(adv_labels)

            if self.args.odr:
                local_adver_images = torch.cat(local_adver_images, 0)
                local_adver_labels = torch.cat(local_adver_labels, 0)
                local_images = torch.cat(local_images, 0)
                local_labels = torch.cat(local_labels, 0)

            for id_, idx in enumerate(sele_users):
                if self.args.odr:
                    log_probs = self.local_models[idx](local_images)[0]
                    loss_list = self.nll_loss_func(self.log_softmax_func(log_probs), local_labels)
                    adv_log_probs = self.local_models[idx](local_adver_images)[0]
                    adv_loss_list = self.nll_loss_func(self.log_softmax_func(adv_log_probs), local_adver_labels)
                    try:
                        if self.args.select:
                            loss_list_final = torch.stack([loss_list[idx] for idx in range(len(adv_loss_list)) if adv_loss_list[idx] > loss_list[idx]])
                        else:
                            loss_list_final = adv_loss_list
                        loss_list_final = torch.mean(loss_list_final)
                        loss_user = local_losses[id_] + self.args.alpha_odr * local_adver_losses[id_] + self.args.beta_odr * torch.mean(loss_list_final)
                    except:
                        loss_user = local_losses[id_]
                else:
                    loss_user = local_losses[id_]

                self.local_optims[idx].zero_grad()
                loss_user.backward()
                self.local_optims[idx].step()

            self.logger.info("{} train: epoch: {}, loss: {}, acc: {}, auc: {}".format(self.args.train_mode, epoch, torch.mean(torch.stack(local_losses)).item(), np.mean(local_accs), np.mean(local_aucs)))

            self.global_epoch+=1
            if self.global_epoch % self.args.eval_epoch == 0:
                self.test(test_global_model = False)

            

    def test(self, test_global_model = False, test_user_set = []):
        if test_global_model:
            tmp_test_mode = "global"
            self.global_model.eval()
        else:
            tmp_test_mode = "local"
            for idx in range(self.args.num_users):
                self.local_models[idx].eval()
        if type(test_user_set) != list:
            test_user_set = [test_user_set]
        elif len(test_user_set) == 0:
            test_user_set = range(self.args.num_users)
        if self.global_epoch not in self.pickle_record["test"].keys():
            self.pickle_record["test"][self.global_epoch] = {}
        if tmp_test_mode not in self.pickle_record["test"][self.global_epoch]:
            self.pickle_record["test"][self.global_epoch][tmp_test_mode] = {}

        with torch.no_grad():
            local_accs = []
            local_aucs = []
            local_losses = []
            for  user in test_user_set:
                self.pickle_record["test"][self.global_epoch][tmp_test_mode][user] = {}
                images, labels = self.get_input( user, train = False)

                if self.args.dataset == "adult"  or self.args.dataset == "eicu":
                    images, labels = images.float().to(self.args.device), labels.float().to(self.args.device)
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                if test_global_model:
                    log_probs = self.global_model(images)[0]
                else:
                    log_probs = self.local_models[user](images)[0]
                loss =  torch.mean(self.nll_loss_func(self.log_softmax_func(log_probs), labels)).item()
                acc, auc = self.acc_auc(log_probs, labels)
                local_accs.append(acc)
                local_aucs.append(auc)
                local_losses.append(loss)
                self.pickle_record["test"][self.global_epoch][tmp_test_mode][user]["loss"] = loss
                self.pickle_record["test"][self.global_epoch][tmp_test_mode][user]["acc"] = acc 
                self.pickle_record["test"][self.global_epoch][tmp_test_mode][user]["auc"] = auc 

        if test_global_model:
            self.global_model.train()
        else:
            for idx in range(self.args.num_users):
                self.local_models[idx].train()
        if self.args.dataset == "adult":
            self.logger.info("target_user: {}, global_epoch: {}, mean acc: {}, mean auc: {}, acc: {}, auc: {}".format(user, self.global_epoch, np.mean(local_accs), np.mean(local_aucs), local_accs, local_aucs ))
        else:
            self.logger.info("target_user: {}, global_epoch: {}, mean acc: {}, mean auc: {}".format(user, self.global_epoch, np.mean(local_accs), np.mean(local_aucs) ))


    def train(self):
        if self.args.train_mode == "global":
            self.global_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch
            self.model_save(model_type = "global", model_name = "global_" + str(self.global_epoch))
            # for user in range(min(100,self.args.num_users)):
            #     self.args.target_user = user 
            #     self.global_epoch = tmp_global_epoch
            #     self.local_train()

        elif self.args.train_mode == "our" and self.args.dataset == "eicu":
            for user in range(self.args.num_users):
                self.local_models[user] = copy.deepcopy(self.global_model)
                self.local_optims[user] = torch.optim.SGD(self.local_models[user].parameters(), lr=self.args.lr_local, momentum=0.5)
            self.odr_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch            
            for user in range(min(100, self.args.num_users)):
                self.args.target_user = user
                self.global_epoch = tmp_global_epoch
                self.local_train()
        
        elif self.args.train_mode == "our" and self.args.dataset == "adult":
            for user in range(self.args.num_users):
                self.local_models[user] = copy.deepcopy(self.global_model)
                self.local_optims[user] = torch.optim.SGD(self.local_models[user].parameters(), lr=self.args.lr_local, momentum=0.5)
            self.odr_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch            
            for user in range(min(100, self.args.num_users)):
                self.args.target_user = user
                self.global_epoch = tmp_global_epoch
                self.local_train()

        elif self.args.train_mode == "our" and self.args.dataset == "cifar10":
            self.odr_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch            
            for user in range(min(100,self.args.num_users)):
                self.args.target_user = user 
                self.global_epoch = tmp_global_epoch
                self.local_train()              

        elif self.args.train_mode == "our" and self.args.dataset == "femnist":
            self.odr_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch            
            for user in range(min(50,self.args.num_users)):
                self.args.target_user = user 
                self.global_epoch = tmp_global_epoch
                self.local_train()  

        elif self.args.train_mode == "our" and self.args.dataset == "cifar100":
            self.odr_train()
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch            
            for user in range(min(100,self.args.num_users)):
                self.args.target_user = user 
                self.global_epoch = tmp_global_epoch
                self.local_train()

        elif self.args.dataset == "eicu" and self.args.train_mode == "g-at-l":
            self.global_train()
            self.local_train()
        elif self.args.dataset == "eicu" and self.args.train_mode == "local":
            self.local_train()
        else:
            tmp_global_epoch = 0
            tmp_global_epoch = self.global_epoch
            for user in range(min(15,self.args.num_users)):
                self.args.target_user = user 
                self.global_epoch = tmp_global_epoch
                self.local_train()   
            
            all_acc = [self.pickle_record["test"][self.global_epoch]["local"][user]["acc"] for user in self.pickle_record["test"][self.global_epoch]["local"].keys()]
            self.logger.info("dataset: {}, num_users: {}, mean acc: {}".format(self.args.dataset, self.args.num_users, np.mean(all_acc)))
