"""
This code allows you to evaluate performance of TA^2-Net with NCC
on the test splits of all datasets (ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, 
vgg_flower, traffic_sign, mscoco, mnist, cifar10, cifar100). 
"""

import os
from torch import nn
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from utils import check_dir
import torch.nn.functional as F

from models.losses import prototype_loss, knn_loss, lr_loss, scm_loss, svm_loss, cross_entropy_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from models.tsa import env_train, val_te_train, resnet_env_sam, resnet_env_max, resnet_eval,resnet_agent
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader, TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES,ALL_METADATASET_NAMES_1)
from config import args, BATCHSIZES
from models.losses import compute_prototypes, prototype_loss
import random


def get_reward(max_action, sample_action, sample):
    context_images = sample['context_images']
    context_labels = sample['context_labels']
    target_labels = sample['target_labels']

    model_6 = get_model(None, args)
    checkpointer = CheckPointer(args, model_6, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model_6 = resnet_env_max(model_6, max_action)
    model_6.cuda()

    model_6.reset()
    env_train(context_images, context_labels, model_6, max_iter=30, lr= 0.5)
    model_7 = get_model(None, args)
    checkpointer = CheckPointer(args, model_7, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model_7 = resnet_env_sam(model_7, sample_action)
    model_7.cuda()
    model_7.reset()
    env_train(context_images, context_labels, model_7,  max_iter=10, lr= 0.5)
                
                
    with torch.no_grad():

                
        model_6.eval()
        context_features = model_6.embed(sample['context_images'])
        target_features = model_6.embed(sample['target_images'])

        _, stat, _ = prototype_loss(context_features, context_labels,
                    target_features, target_labels)


        model_7.eval()
        context_features_7 = model_7.embed(sample['context_images'])
        target_features_7 = model_7.embed(sample['target_images'])


        _, stat_7, _ = prototype_loss(context_features_7, context_labels,
                    target_features_7, target_labels)


    reward = (stat_7['acc'] - stat['acc'])*100
    
    return reward

def meta_val(model_agent, valsets, trainsets, val_loader, session):
    epoch_val_loss = {name: [] for name in valsets}
    epoch_val_acc = {name: [] for name in valsets}
    dataset_accs, dataset_losses = [], []
    for valset in valsets:
        if valset in trainsets:
            lr = 0.05
            lr_beta = 0.1
        else:
            lr = 2
            lr_beta = 1

        val_losses, val_accs = [], []
        for j in tqdm(range(args['train.eval_size'])):
            
            sample = val_loader.get_validation_task(session, valset)
            context_labels = sample['context_labels']
            n_way = len(context_labels.unique())
            _, max_action, _, _  = model_agent.embed(sample['context_images'], context_labels,  n_way)
            model_val = get_model(None, args)
            checkpointer = CheckPointer(args, model_val, optimizer=None)
            checkpointer.restore_model(ckpt='best', strict=False)
            model_val = resnet_eval(model_val, max_action)
            model_val.cuda()

            model_val.reset()
            val_te_train(sample['context_images'], context_labels, model_val, max_iter=40, lr=lr, lr_beta=lr_beta)

            with torch.no_grad():
                context_features = model_val.embed(sample['context_images'])
                target_features = model_val.embed(sample['target_images'])

                context_features = model_val.beta(context_features)
                target_features = model_val.beta(target_features)

                context_labels = sample['context_labels']
                target_labels = sample['target_labels']

                _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                    target_features, target_labels)
            val_losses.append(stats_dict['loss'])
            val_accs.append(stats_dict['acc'])

        # write summaries per validation set
        dataset_acc, dataset_loss = np.mean(val_accs) * 100, np.mean(val_losses)
        dataset_accs.append(dataset_acc)
        dataset_losses.append(dataset_loss)
        epoch_val_loss[valset].append(dataset_loss)
        epoch_val_acc[valset].append(dataset_acc)
        print(f"{valset}: val_acc {dataset_acc:.2f}%, val_loss {dataset_loss:.3f}")

    # write summaries averaged over datasets
    avg_val_loss, avg_val_acc = np.mean(dataset_losses), np.mean(dataset_accs)

    return avg_val_loss, avg_val_acc, epoch_val_loss, epoch_val_acc


def test(trainsets, model_agent):

    TEST_SIZE = 600

    testsets = ALL_METADATASET_NAMES_1 # comment this line to test the model on args['data.test']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])

    accs_names = ['NCC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            if dataset in trainsets:
                lr = 0.05
                lr_beta = 0.1
            else:
                lr = 0.5
                lr_beta = 1

            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):

                sample = test_loader.get_test_task(session, dataset)
                context_labels = sample['context_labels']
                target_labels = sample['target_labels']
                n_way = len(context_labels.unique())
                _, max_action, _, _  = model_agent.embed(sample['context_images'], context_labels,  n_way)


                model_val = get_model(None, args)
                checkpointer = CheckPointer(args, model_val, optimizer=None)
                checkpointer.restore_model(ckpt='best', strict=False)
                model_val = resnet_eval(model_val, max_action)
                model_val.cuda()

                model_val.reset()
                val_te_train(sample['context_images'], context_labels, model_val, max_iter=40, lr=lr, lr_beta=lr_beta)

                with torch.no_grad():
                    context_features = model_val.embed(sample['context_images'])
                    target_features = model_val.embed(sample['target_images'])

                    context_features = model_val.beta(context_features)
                    target_features = model_val.beta(target_features)

                _, stats_dict, _ = prototype_loss(context_features, context_labels,
                                                    target_features, target_labels)
                var_accs[dataset]['NCC'].append(stats_dict['acc'])
            dataset_acc = np.array(var_accs[dataset]['NCC']) * 100
            print(f"{dataset}: test_acc {dataset_acc.mean():.2f}%")
    # Print nice results table
    print('results of {} with {}'.format(args['model.name'], args['test.tsa_opt']))
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)
    out_path = os.path.join(args['out.dir'], 'weights')
    out_path = check_dir(out_path, True)
    out_path = os.path.join(out_path, '{}-tsa-{}-{}-{}-{}-test-results-.npy'.format(args['model.name'], args['test.tsa_opt'], args['test.tsa_ad_type'], args['test.tsa_ad_form'], args['test.tsa_init']))
    np.save(out_path, {'rows': rows})

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")

def main():
    TEST_SIZE = 1000000

    best_val_loss = 999999999
    best_val_acc = 0
    
    # Setting up datasets
    if args['test.mode'] == 'mdl':
        # multi-domain learning setting, meta-train on 8 training sets
        trainsets = TRAIN_METADATASET_NAMES
        valsets = TRAIN_METADATASET_NAMES

    elif args['test.mode'] == 'sdl':
        # single-domain learning setting, meta-train on ImageNet
        trainsets = ['ilsvrc_2012']
        valsets = ['ilsvrc_2012']



    train_loader = MetaDatasetEpisodeReader('train', trainsets)

    val_loader = MetaDatasetEpisodeReader('val', trainsets, valsets)

    model_agent = get_model(None, args)
    checkpointer = CheckPointer(args, model_agent, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model_agent.eval()
    model_agent = resnet_agent(model_agent)
    model_agent.reset()
    model_agent.cuda()
    
    params_1 = [v for k, v in model_agent.named_parameters() if 'conv_fc' in k or 'bn_fc' in k]
    params_2 = [v for k, v in model_agent.named_parameters() if 'alpha' in k or 'garm' in k or 'rarm' in k]


    params = []
    params.append({'params': params_1})
    params.append({'params': params_2})
    
    optimizer_gate = torch.optim.Adam(params,lr=0.0001)


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False

    
    with tf.compat.v1.Session(config=config) as session:
        
            loss_all = 0.0 
            for k in tqdm(range(TEST_SIZE)):

                
                # loading a task containing a support set and a query set
                sample = train_loader.get_train_task(session)
                context_labels = sample['context_labels']

                n_way = len(context_labels.unique())
            
                context_features, max_action, sample_action, loss  = model_agent.embed(sample['context_images'], context_labels,  n_way)
                context_features = context_features.flatten(1)
                loss_model, _, _ = prototype_loss(context_features, context_labels,
                                       context_features, context_labels)
                
                
                reward = get_reward(max_action, sample_action, sample)
                
                loss = - loss.sum() *reward *0.01 + loss_model 


                loss_all = loss_all + loss

                
                if (k+1) % len(trainsets) ==0 :

                    loss_all = loss_all / len(trainsets)

                    optimizer_gate.zero_grad()
                
                    loss_all.backward()

                    optimizer_gate.step()
                    loss_all = 0.0

                if (k + 1) % args['train.eval_freq'] == 0:

                    model_agent.eval()
                    avg_val_loss, avg_val_acc, epoch_val_loss, epoch_val_acc = meta_val(model_agent, valsets, trainsets, val_loader, session)

                    if avg_val_acc > best_val_acc:
                        best_val_loss, best_val_acc = avg_val_loss, avg_val_acc
                        is_best = True
                        print('Best model so far!')
                    else:
                        is_best = False
                    extra_dict = {'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc}
                    checkpointer.save_checkpoint(k, best_val_acc, best_val_loss,
                                             is_best, 
                                             state_dict=model_agent.get_state_dict(), extra=extra_dict)

                    model_agent.train()
        
    test(trainsets, model_agent)
                    


if __name__ == '__main__':
    main()



