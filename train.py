import argparse, os, copy, glob, datetime, sys
from email.errors import HeaderParseError
from email.policy import default
from collections import defaultdict
from genericpath import exists

# import nbformat.v1.rwbase
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import random
import dsmil as mil

from utils import Logger
from dataset.general_tcga import Gneral_TCGA
from dataset.general_camelyon import Gneral_CAMELYON
from dataset.loader import dataloader, collate_fn
from sklearn.metrics import roc_curve, roc_auc_score
from hm import HybridMemory
import torch.nn.functional as F
from collections import defaultdict


def get_bag_labels_ori(labels, num_classes):  
    label_one_hot_real = torch.zeros(labels.size(0), num_classes)
    for i in range(labels.size(0)):
        label_one_hot_real[i, labels[i]] = 1
    return label_one_hot_real


def train(train_df, milnet, criterion_BCELL, criterion_CE,  optimizer, num_classes, args, epoch, memory):
    milnet.train()
    total_loss = 0

    for i, data in enumerate(train_df):
        optimizer.zero_grad()
        feats, label_ori, lengths = data
        label = get_bag_labels_ori(label_ori, num_classes)

        bag_label, bag_feats = label.cuda(), feats.cuda()  
        
        # classes, prediction_bag_conv, prediction_bag_fc, feats
        ins_prediction, bag_prediction_conv, bag_prediction_fc, output_feats = milnet(bag_feats)
        
        # for contrastive loss
        output_feats = F.normalize(output_feats, dim=-1)
        contrastive_loss = memory(output_feats[:, int(label_ori), :], label_ori.cuda(), epoch)

        # for instance loss and bag loss
        max_prediction, _ = torch.max(ins_prediction, 0) 
        bag_loss =  criterion_CE(max_prediction[None,], bag_label) + criterion_CE(bag_prediction_fc, bag_label) 

        # overall loss
        loss = bag_loss  + 0.5 *  contrastive_loss

        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

    return total_loss / len(train_df)


def test(test_df, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args):
    milnet.eval()

    total_loss = 0
    test_labels = []
    test_predictions = []
    right, count = 0, 0
    dic_num = defaultdict(int)
    dic_right_num = defaultdict(int)
    with torch.no_grad():
        for i, data in enumerate(test_df):
            feats, label_ori, lengths = data
            label = get_bag_labels_ori(label_ori, num_classes)

            num_slides = len(lengths)
            bag_label, bag_feats = label.cuda(), feats.cuda()
            ins_prediction, bag_prediction_conv, bag_prediction_fc, feats = milnet(bag_feats)
            
            # for loss calculation
            max_prediction, _ = torch.max(ins_prediction, 0)   
            loss = criterion_CE(max_prediction[None,], bag_label) + criterion_CE(bag_prediction_fc, bag_label) 
            total_loss = total_loss + loss.item()

            test_labels.extend([label])
            test_prediction = F.softmax(bag_prediction_fc, dim=1).cpu()
            # test_predictions.extend([torch.sigmoid(bag_prediction).cpu()])
            temp = torch.argmax(test_prediction).cpu()
            
            dic_num[int(label_ori)] += 1
            if temp == label_ori:
                right +=1
                dic_right_num[int(label_ori)] += 1
    
            test_predictions.extend([test_prediction])
        
        count = i + 1

    for key in sorted(dic_num.keys()):
        print('Accuracy of class {} is {}'.format(key, dic_right_num[key]/dic_num[key]))

    test_labels = torch.cat(test_labels, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, num_classes, pos_label=1)

    for i in range(num_classes):
        class_prediction_bag = copy.deepcopy(test_predictions[:, i])
        class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
        class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
        test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i].squeeze()) + bag_score
  
    avg_score = right / count
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    # must check before training
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')

    parser.add_argument('--split_train', default=0.65, type=float, help='Training/Validation split [0.5]')
    parser.add_argument('--split_val', default=0.15, type=float, help='Training/Validation split [0.5]')
    parser.add_argument('--eva_epoch', default=1, type=int)
    ###################################
    parser.add_argument('--drop_p', default=0.1, type=float, help='drop portion during training')
    parser.add_argument('--drop_probability', default=0.4, type=float, help='drop portion during training') 
    parser.add_argument('--scheduler', default='cos', type=str, help='type of schedular')
    ###################################
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    ####################################
    parser.add_argument('--temp', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.2, type=float, help='0.2')

    parser.add_argument('--margin', default=-0.1, type=float,)  # 0.2 for Cam
    parser.add_argument('--p1', default=0.05, type=float, help='for cam, 0.01') 
    parser.add_argument('--p2', default=0.01, type=float, help='for cam 0.05') 
    ####################################

    parser.add_argument('--save_dir', default='logs', type=str)
    # parser.add_argument('--data_dir', default='/Workspace/zhikangwang/Datasets/TCGA/Kidney', type=str)
    # parser.add_argument('--data_dir', default='/Workspace/zhikangwang/Datasets/TCGA/Lung', type=str)
    parser.add_argument('--data_dir', default='/Workspace/zhikangwang/Datasets/CAMELYON/CAMELYON16', type=str)
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # for reproduction
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # for log save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    sys.stdout = Logger(os.path.join(args.save_dir, 'log_train' + str(len(os.listdir(args.save_dir))+1) + '.txt'))

    # generate dataset and dataloader
    if 'TCGA' in args.data_dir:
        dataset = Gneral_TCGA(root=args.data_dir, portion_train=args.split_train, portion_val=args.split_val)
    else:
        dataset = Gneral_CAMELYON(root=args.data_dir, portion_train=args.split_train, portion_val=args.split_val)

    train_dataloader = DataLoader(dataloader(dataset=dataset.train, training=True, drop_p=args.drop_p, drop_probability=args.drop_probability),
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  pin_memory=True)

    val_dataloader = DataLoader(dataloader(dataset=dataset.val, training=False),
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn,
                                shuffle=False,
                                pin_memory=True)

    test_dataloader = DataLoader(dataloader(dataset=dataset.test, training=False),
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 pin_memory=True)

    # generate network, optimizers
    if 'CAMELYON' in args.data_dir:
        CAMEYLON = True
    else: 
        CAMEYLON = False
    print('CAMEYLON is {}'.format(CAMEYLON))
    num_classes = dataset.num_classes
    i_classifier = mil.FCLayer(in_size=args.feats_size * 2, out_size=num_classes).cuda()
    milnet = mil.MILNet(i_classifier, num_classes, args.margin, args.p1, args.p2, CAMEYLON).cuda()

    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion_BCELL = nn.BCEWithLogitsLoss()
    criterion_CE = nn.CrossEntropyLoss()

    # modify save path
    save_path = os.path.join('weights', datetime.date.today().strftime("%m%d%Y"))
    os.makedirs(save_path, exist_ok=True)

    # for memory bank
    memory = HybridMemory(512, num_classes, temp=args.temp, momentum=args.momentum)
    # from collections import defaultdict
    # features = defaultdict(list)
    # with torch.no_grad():
    #     milnet.eval()
    #     for data in train_dataloader:
    #         feats, label_ori, lengths = data
    #         index = int(label_ori)
    #         bag_feats = feats.cuda()
    #         ins_prediction, bag_prediction, output_feats, _ = milnet(bag_feats)
    #         features[index].append(output_feats[:, index, :])
    #
    #     features = [torch.cat(features[key], dim=0).mean(dim=0, keepdim=True) for key in features.keys()]
    #     features = torch.cat(features, dim=0)
    #
    # memory.features = F.normalize(features, dim=1)
    memory.labels = torch.arange(num_classes).cuda()

    # start training and testing
    best_score_val = 0.0
    best_score_test = 0.0
    for epoch in range(1, args.num_epochs + 1):
        train_loss_bag = train(train_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args, epoch, memory)
        lr = scheduler.get_last_lr()
        print('Epoch {}/{}: average loss ->{:4f}  lr ->{}'.format(epoch, args.num_epochs, train_loss_bag, lr))
        scheduler.step()
        
        if epoch % args.eva_epoch == 0:
            test_loss_bag, avg_score, aucs, _ = test(val_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args)
            print('Val Set: Epoch [%d/%d] test loss: %.4f, average score: %.4f, AUC: ' %(epoch, args.num_epochs, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))
            
            current_score = sum(aucs)/len(aucs) + avg_score
            if current_score > best_score_val:
                best_score_val = current_score
                save_name = os.path.join(save_path, 'best_model_val.pth')
                torch.save(milnet.state_dict(), save_name)
                print('Get best val Model')

            test_loss_bag, avg_score, aucs, _ = test(test_dataloader, milnet, criterion_BCELL, criterion_CE, optimizer, num_classes, args)
            print('Testing Set: Epoch [%d/%d] test loss: %.4f, average score: %.4f, AUC: ' %(epoch, args.num_epochs, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs)))
            
            current_score = sum(aucs)/len(aucs) + avg_score
            if current_score > best_score_test:
                best_score_test = current_score
                save_name = os.path.join(save_path, 'best_model_test.pth')
                torch.save(milnet.state_dict(), save_name)
                print('Get best test Model')

        print('\n')    

if __name__ == '__main__':
    main()
