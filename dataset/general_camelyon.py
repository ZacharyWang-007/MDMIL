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


class Gneral_CAMELYON(object):
    def __init__(self, root, portion_train, portion_val, **kwargs):
        super(Gneral_CAMELYON, self).__init__()
        self.root = root

        self.label_dic = {'CLAM_tumor': 0, 'CLAM_normal': 1}

        train_path, val_path = [], []
        for key in self.label_dic.keys():
            temp = glob.glob(os.path.join(self.root, 'training', key, 'pt_files', '*.pt'))
            length = len(temp)
            train_path += temp[0: int(length * 0.9)]
            val_path += temp[int(length * 0.9): ]
            # val_path += temp[0: int(length * 0.9): ]
        
        # random.shuffle(train_path)

        test = []
        csv = pd.read_csv(os.path.join(root, 'testing', 'reference.csv'))
        datas = csv.values
        data_analysis_test = defaultdict(int)
        for i in datas:
            if i[1] == 'Normal':
                label = 1
            else:
                label = 0
            
            # print(label)
            test.append((os.path.join(root, 'testing', 'CLAM_images', 'pt_files', i[0] + '.pt'), label))
            data_analysis_test[label] += 1
            
        
        self.train, data_analysis_train = self._process_dir(train_path)
        self.val, data_analysis_val = self._process_dir(val_path)
        self.test = test
        # self.test, data_analysis_test = self._process_dir(test_path)

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

            data_analysis[label] +=1
            dataset.append((file, label))
                
        return dataset, data_analysis
