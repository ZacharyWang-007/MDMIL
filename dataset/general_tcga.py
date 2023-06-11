import fractions
import glob
from json.tool import main
import re

import os
import random
import os.path as osp
from collections import defaultdict
import pandas as pd
# from sklearn.utils import shuffle


class Gneral_TCGA(object):
    def __init__(self, root, portion_train, portion_val, **kwargs):
        super(Gneral_TCGA, self).__init__()
        self.root = root
        # bags_path = glob.glob(os.path.join(self.root, 'CLAM_*', 'pt_files', '*.pt'))
        # random.shuffle(bags_path)

        # length = len(bags_path)

        # step1, step2 = int(length * portion_train), int(length * portion_train + length * portion_val)

        self.label_dic = {'CLAM_LUAD': 0, 'CLAM_LUSC': 1, 'CLAM_KICH': 0, 'CLAM_KIRC': 1, 'CLAM_KIRP': 2}

        train_path, val_path, test_path = [], [], []
        for key in self.label_dic.keys():
            temp = glob.glob(os.path.join(self.root, key, 'pt_files', '*.pt'))
            length = len(temp)
            train_path += temp[0: int(length * portion_train)]
            val_path += temp[int(length * portion_train): int(length * (portion_train + portion_val))]
            test_path += temp[int(length * (portion_train + portion_val)): ]

        # train_path = bags_path[0: step1]
        # val_path = bags_path[step1: step2]
        # test_path = bags_path[step2: ]

        self.train, data_analysis_train = self._process_dir(train_path)
        self.val, data_analysis_val = self._process_dir(val_path)
        self.test, data_analysis_test = self._process_dir(test_path)

        self.num_classes = len(data_analysis_train.keys())
        
        for key in sorted(data_analysis_train.keys()):
            print('--------------------------')
            print('Label {}: Training Set  has {} samples, Val Set has {} samples Testing Set has {} samples'
                  .format(key, data_analysis_train[key], data_analysis_val[key], data_analysis_test[key]))
        print('--------------------------')

    def _process_dir(self, files):
        dataset = []
        data_analysis = defaultdict(int)
        for file in files:
            label = self.label_dic[file.split('/')[-3]]

            if int(file.split('/')[-1].split('-')[3][0:2]) > 10: continue

            data_analysis[label] +=1
            dataset.append((file, label))
                
        return dataset, data_analysis


if __name__ == '__main__':
    # Gneral_TCGA(root='/Workspace/zhikangwang/Datasets/TCGA_test/datasets/tcga-dataset', csv_file_df='TCGA.csv', partition=0.3)
    dataset = Gneral_TCGA(root='/Workspace/zhikangwang/Datasets/TCGA_test/datasets/Camelyon16', csv_file_df='Camelyon16.csv', partition=0.3)