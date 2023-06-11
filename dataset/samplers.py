from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        for index, (path, label, _) in enumerate(data_source):
            folder_name = path.split('/')[-2]
            self.index_dic[folder_name].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            # if len(t) >= self.num_instances:
            #     t = np.random.choice(t, size=self.num_instances, replace=False)
            # else:
            #     t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        

        return iter(ret)