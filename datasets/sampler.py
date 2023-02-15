import torch
import numpy as np
import random
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler

class Balanced_sampler(Sampler):
    '''
    Deterministic balanced sampler for each class
    '''
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.count = 0
        self.bs = self.n_samples * self.n_classes
        self.label_list = dataset.tgts
        self.label_set = list(set(self.label_list))
        self.label_set.sort()
        self.label_to_indices = {label: np.where(np.array(self.label_list) == label)[0] for label in self.label_set}
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.label_set}

    def __iter__(self):
        self.count = 0
        while self.count + self.bs < len(self.label_list):
            indices = []
            for class_ in self.label_set:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_] : self.used_label_indices_count[class_]+self.n_samples].tolist())
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # random.shuffle(indices)
            yield indices
            self.count += self.bs

    def __len__(self):
        return len(self.label_list) // self.bs

class Random_Balanced_sampler(Sampler):
    '''
    Ramdomly sampled balanced categorical sampler
    '''
    def __init__(self, dataset, n_classes, n_samples, limit=128):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.count = 0
        self.limit = limit
        if self.n_samples * self.n_classes <= limit:
            self.bs = self.n_samples * self.n_classes
        else:
            self.bs = int(limit/self.n_samples)*self.n_samples
        self.label_list = dataset.tgts
        self.label_set = list(set(self.label_list))
        self.label_set.sort()
        self.label_to_indices = {label: np.where(np.array(self.label_list) == label)[0] for label in self.label_set}
        for l in self.label_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.label_set}

    def __iter__(self):
        self.count = 0
        while self.count + self.bs < len(self.label_list):
            indices = []

            if self.n_samples * self.n_classes > self.limit:
                selected_classes = random.sample(self.label_set, int(self.limit/self.n_samples))
            else:
                selected_classes = self.label_set
                
            for class_ in selected_classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_] : self.used_label_indices_count[class_]+self.n_samples].tolist())
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # random.shuffle(indices)
            yield indices
            self.count += self.bs

    def __len__(self):
        return len(self.label_list) // self.bs



    


