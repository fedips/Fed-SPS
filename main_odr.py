# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_mtl.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
import torch
from torch import nn
import pdb
from utils.train_utils import get_model, get_data, read_data, get_random_dir_name, setup_seed, construct_log
from utils.odr_options import args_parser
from models.OdrUpdate import LocalTrain
from models.test import test_img_local_all, test_img_local
import os
import time
import pickle
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.global_model_dir = os.path.join(args.target_dir, args.global_model_dir)
    args.local_model_dir = os.path.join(args.target_dir, args.local_model_dir)
    setup_seed(seed = args.seed)
    logger = construct_log(args)

    if args.dataset == "cifar10" or args.dataset == "mnist" or args.dataset == "femnist":
        args.num_classes = 10
    elif args.dataset == "adult" :
        args.num_classes = 2
        args.num_users = 2
    elif  args.dataset == "eicu" or args.dataset == "sent140":
        args.num_classes = 2
        args.num_users = 14
    elif args.dataset == "cifar100":
        args.num_classes = 100
    else:
        pass


    if 'cifar' in args.dataset or args.dataset == 'mnist' :
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    elif args.dataset == 'eicu' or args.dataset == 'adult':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)

    else:
        if 'femnist' in args.dataset:
            train_path = os.path.join(args.dataset_dir, "leaf-master/data", args.dataset, "data/mytrain")
            test_path =  os.path.join(args.dataset_dir, "leaf-master/data", args.dataset, "data/mytest") 
        elif "sent140" == args.dataset:
            train_path = os.path.join(args.dataset_dir, "leaf-master/data", args.dataset, "data/train")
            test_path = os.path.join(args.dataset_dir, "leaf-master/data", args.dataset, "data/train") 
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)

        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())

        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    net = get_model(args)
    learning_model = LocalTrain(args, logger, dataset_train, dataset_test, dict_users_train, dict_users_test, net)
    tmp_dir = os.path.split(os.path.realpath(__file__))[0]
    
    if args.auto_deploy:
        os.system("cp -r {} {}".format(tmp_dir, args.target_dir))
        try:
            learning_model.train()
            with open(os.path.join(args.target_dir, "pickle.pkl"), "wb") as f:
                pickle.dump(learning_model.pickle_record, f)
            os.makedirs( os.path.join(args.target_dir, "done"), exist_ok = True)
        except Exception as e:
            logger.info("error info: {}.".format(e))
    else:
        learning_model.train()
        with open(os.path.join(args.target_dir, "pickle.pkl"), "wb") as f:
            pickle.dump(learning_model.pickle_record, f)
        os.makedirs( os.path.join(args.target_dir, "done"), exist_ok = True)

