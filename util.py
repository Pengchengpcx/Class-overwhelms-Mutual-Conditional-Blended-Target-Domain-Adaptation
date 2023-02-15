import os 
import random
import torch
from torch.autograd import Variable
import numpy as np
import itertools
import math
from copy import deepcopy
import shutil
import logging
import glob

MODEL_ROOT = 'systempath'

def delete_model(sub_dir):
    """Delete non-optimal models"""
    file_dir = os.path.join(MODEL_ROOT, sub_dir)
    os.system('rm -rf {}'.format(file_dir))

def save_model(net, sub_dir, filename, logger=None):
    """Save trained model."""
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)
    file_dir = os.path.join(MODEL_ROOT, sub_dir)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    torch.save(net.state_dict(),
               os.path.join(file_dir, filename))
    if logger == None:
        print("save pretrained model to: {}".format(os.path.join(file_dir, filename)))
    else:
        logger.info("save pretrained model to: {}".format(os.path.join(file_dir, filename)))

def load_model(net, sub_dir, filename):
    """Load saved model"""
    path = os.path.join(MODEL_ROOT, sub_dir, filename)
    path = glob.glob(path)
    if os.path.exists(path[0]):
        net.load_state_dict(torch.load(path[0]), strict=False)
    else:
        print('Target file does not exist:',path)
        raise NotImplementedError

    return net

def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def confidence_pseudo_label(loader, metric, margin):
    '''
    Filter of the pseudo labels satisfying the confidence margin

    loader: pseudo label dataloader
    metric: distance metric for pseudo labels, 'cosine' or 'L2'
    margin: margin to filter confident pseudo labels
    '''
    count = 0
    for step, (_,labs) in enumerate(loader):
        bs = labs.shape[0]
        labs = labs.cuda()
        if metric == 'cosine':
            continue
        elif metric == 'L2':
            labs = -labs
        else:
            print('Do not support the metric')
            raise NotImplementedError
        
            confidence, index = torch.topk(labs, 2, dim=1)
            value = confidence[:,0] - confidence[:,1]
            acc = torch.sum(value>margin)
            count += acc.cpu().item()
    
    return step*bs, count



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def to_onehot(label, class_num):
    bs = label.shape[0]
    label = label.reshape(-1,1)
    if label.is_cuda == True:
        onehot = torch.zeros(bs, class_num).cuda().scatter_(1, label, 1)
    else:
        onehot = torch.zeros(bs, class_num).scatter_(1, label, 1)
    return onehot

def adaptive_margin2(hi, lo, alpha, max_iter, iter_num):

    margin = np.float(2.0 * (hi - lo) / (1.0 + np.exp(-alpha * iter_num / max_iter))
            - (hi - lo) + lo)

    return margin

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class Scaling_coeff:
    def __init__(self, alpha=1.0, lo=0, hi=1, max_iters=1000):
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.max_iters = max_iters
        self.iter_num = 0
    
    def step(self):
        coeff = np.float(2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        self.iter_num += 1

        return coeff
