import os
from PIL import Image
import warnings
from RandAugment import RandAugment

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import bisect
from datasets.sampler import Balanced_sampler, Random_Balanced_sampler

from torchvision import transforms
import torch

class ResizeImage:
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size

    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

image_train = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

image_test = transforms.Compose([
        ResizeImage(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

rand_augmentation = transforms.Compose([
      		#transforms.Resize((256, 256)), # spatial size of vgg-f input
            transforms.Resize(256),
      		transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(1, 2.0),
      		transforms.ToTensor(),
      		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])

class ResizeImage:
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size

    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))


class PlaceCrop:
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

"""
Code adapted from CDAN github repository.
https://github.com/thuml/CDAN/tree/master/pytorch
"""


class ImageList(Dataset):
    def __init__(self, image_root, image_list_root, dataset, domain_label, dataset_name, split='train', transform=None, 
                 sample_masks=None, pseudo_labels=None, strong_transform=None, aug_num=0, rand_aug=False):
        self.image_root = image_root
        self.dataset = dataset  # name of the domain
        self.dataset_name = dataset_name  # name of whole dataset
        self.transform = transform
        self.strong_transform = strong_transform
        self.loader = self._rgb_loader
        self.sample_masks = sample_masks
        self.pseudo_labels = pseudo_labels
        self.rand_aug = rand_aug
        self.aug_num = aug_num
        if dataset_name == 'domainnet' or dataset_name == 'minidomainnet':
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '_' + split + '.txt'), domain_label)
        else:
            imgs = self._make_dataset(os.path.join(image_list_root, dataset + '.txt'), domain_label)
        self.imgs = imgs
        self.tgts = [s[1] for s in imgs]
        if sample_masks is not None:
            temp_list = self.imgs
            self.imgs = [temp_list[i] for i in self.sample_masks]
            if pseudo_labels is not None:
                self.labels = self.pseudo_labels[self.sample_masks]
                assert len(self.labels) == len(self.imgs), 'Lengths do no match!'

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path, domain):
        image_list = open(image_list_path).readlines()
        images = [(val.split()[0], int(val.split()[1]), int(domain)) for val in image_list]
        return images

    def __getitem__(self, index):
        output = {}
        path, target, domain = self.imgs[index]
        if self.dataset_name == 'domainnet' or self.dataset_name == 'minidomainnet':
            raw_img = self.loader(os.path.join(self.image_root, path))

        elif self.dataset_name in ['office-home', 'office31', 'office-home-lds', 'office-home-btlds']:
            raw_img = self.loader(os.path.join(self.image_root, path))

        if self.transform is not None:
            img = self.transform(raw_img)
        if self.rand_aug and self.strong_transform!= None:
            aug_img = [self.strong_transform(raw_img) for i in range(self.aug_num)]
            output['strong_img'] = aug_img
        
        output['img'] = img
        if self.pseudo_labels is not None:
            output['target'] = torch.squeeze(torch.LongTensor([np.int64(self.labels[index]).item()]))
        else:
            output['target'] = torch.squeeze(torch.LongTensor([np.int64(target).item()]))
        output['domain'] = domain
        output['idx'] = index

        return output

    def __len__(self):
        return len(self.imgs)

def build_dataset(args, data_name, source_name, target_name, bs, catal, num_workers, aug_num=0, rand_aug=False):
    '''
    Build the source and multiple targets dataloaders
    data_name: dataset name
    source: source domain name
    bs: batch_size
    catal: categorical dataloaders
    rand_aug: Rand Aug

    Note: Except that DomainNet follows the inductive setting, others follow transductive setting
    '''
    all_domains = {
        'office31': 
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['amazon', 'dslr', 'webcam'],
                'numbers':[2817, 498, 795],
                'classes': 31,
                'neurons': 128
            },
        'office-home':
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['Art', 'Clipart', 'Product', 'Real_World'],
                'numbers':[2427, 4365, 4439, 4357],
                'classes': 65,
                'neurons': 128,
                'LDS_split_path':'/'
            },
        'office-home-rsut':
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['Clipart_RS', 'Clipart_UT', 'Product_RS', 'Product_UT', 'Real_World_RS', 'Real_World_UT'],
                'numbers': 'it different depends on RS, UT, BS, BT',
                'classes': 65,
                'neurons': 128,
            },
        'visda-c':
            {
                'path': '/',
                'sub_domains': ['train', 'validation'],
                'numbers': [152397, 55388],
                'classes': 12,
                'neurons': 128
            },
        'imageclef':
            {   
                'path': '/',
                'list_root': 'data/imageclef/',
                'sub_domains': ['b', 'c', 'i', 'p'],
                'numbers':[600, 600, 600, 600],
                'classes': 12,
                'neurons': 128,
            },
        'domainnet':
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                'numbers':[14604, 15582, 21850, 51750, 52041, 20916],
                'classes': 345,
                'neurons': 128
            },
        'minidomainnet':
            {
                'path': '/',
                'list_root': '/',
                'sub_domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                'numbers':[1461, 1559 , 2185, 5175, 5205, 2092],
                'classes': 345,
                'neurons': 128
            },
    }

    # set single source and others as targets
    args.num_classes = all_domains[data_name]['classes']
    args.num_domains = len(all_domains[data_name]['sub_domains'])

    if args.source == args.target:
        raise ValueError('source domain cannot be the same as target domain')
    
    dsets, dset_loaders = {}, {}

    dsets['source_train'] = ImageList(image_root=all_domains[data_name]['path'], image_list_root=all_domains[data_name]['list_root'],
                                dataset=all_domains[data_name]['sub_domains'][source_name], transform=image_train,
                                domain_label=source_name, dataset_name=data_name, split='train', strong_transform=rand_augmentation, aug_num=args.aug_num, rand_aug=args.rand_aug)

    dsets['source_test'] = ImageList(image_root=all_domains[data_name]['path'], image_list_root=all_domains[data_name]['list_root'],
                                dataset=all_domains[data_name]['sub_domains'][source_name], transform=image_test,
                                domain_label=source_name, dataset_name=data_name, split='test')
    
    if args.catal == True:
        sampler = Random_Balanced_sampler(dsets['source_train'], all_domains[data_name]['classes'], bs, args.bs_limit)
        dset_loaders['source_train'] = DataLoader(dsets['source_train'], num_workers=num_workers, pin_memory=False, batch_sampler=sampler,)
    else:
        dset_loaders['source_train'] = DataLoader(dsets['source_train'], batch_size=bs, shuffle=True,
                                            num_workers=num_workers, drop_last=True, pin_memory=False)
    dset_loaders['source_test'] = DataLoader(dataset=dsets['source_test'], batch_size=64, num_workers=num_workers, drop_last=False, pin_memory=False)

    dsets['target_train'] = ImageList(image_root=all_domains[data_name]['path'], image_list_root=all_domains[data_name]['list_root'],
                                dataset=all_domains[data_name]['sub_domains'][target_name], transform=image_train,
                                domain_label=target_name, dataset_name=data_name, split='train', strong_transform=rand_augmentation, aug_num=args.aug_num, rand_aug=args.rand_aug)

    dsets['target_test'] = ImageList(image_root=all_domains[data_name]['path'], image_list_root=all_domains[data_name]['list_root'],
                                dataset=all_domains[data_name]['sub_domains'][target_name], transform=image_test,
                                domain_label=target_name, dataset_name=data_name, split='test')
    # scale batch size
    if args.catal ==True:
        if all_domains[data_name]['classes']*bs > args.bs_limit:
            scale_bs = int(args.bs_limit/bs)*bs
        else:
            scale_bs = all_domains[data_name]['classes']*bs
    else:
        scale_bs = bs

    dset_loaders['target_train'] = DataLoader(dataset=dsets['target_train'], batch_size=scale_bs, shuffle=True,
                                                        num_workers=num_workers, drop_last=True, pin_memory=False)

    dset_loaders['target_test'] = DataLoader(dataset=dsets['target_test'], batch_size=64, num_workers=num_workers, drop_last=False,
                                                        pin_memory=False)

    return dsets, dset_loaders

