path = 'add systempath here'
import sys
sys.path.append(path)
import time
import argparse
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets.preparedata_lds import build_dataset
from torch.optim.lr_scheduler import LambdaLR
from models.lr_scheduler import StepwiseLR
from models.resnet_lowlevel import *
from models.discriminator import *
from models.grl import *
from models.evaluation import cal_accuracy, cal_acc
from util import *
import copy
from torch.cuda.amp import autocast as autocast


class AdvLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, inputs):
        inputs = inputs.softmax(dim=1)
        loss = - torch.log(inputs + self.eps).mean(dim=1)
        return loss.mean()

def main(args):
    # create dataloaders
    dsets, dset_loaders, concate_sets, concate_loaders = build_dataset(args=args, data_name=args.dataset, source_name=args.source, bs=args.batch_size, \
                                                                        catal=args.catal, num_workers=args.workers, aug_num=args.aug_num, rand_aug=args.rand_aug)
    dset_loaders_source_test = copy.deepcopy(dset_loaders['source'])
    dset_loaders['source'] = ForeverDataIterator(dset_loaders['source'])
    concate_loaders['train'] =  ForeverDataIterator(concate_loaders['train'])

    for t in dset_loaders['target_train'].keys():
        dset_loaders['target_train'][t] = ForeverDataIterator(dset_loaders['target_train'][t])

    # create models
    if args.net == 'resnet50':
        encoder = resnet50(args, pretrained=True).cuda()
    elif args.net == 'resnet101':
        encoder = resnet101(args, pretrained=True).cuda()

    discriminator = CatDomain(args.feat_dim, args.hid_dim, args.num_classes).cuda()
    args.discriminator = discriminator.name
    name = args.dataset + dsets['source'].dataset

    # get logger
    logger_path = path + 'logs/{}/{}_{}.log'.format(args.dataset, dsets['source'].dataset, args.sub_log)
    logger = get_logger(logger_path)
    logger.info(args)
    logger.info('Start training process')

    # set optimizer
    resent_params = [
    {'params': encoder.conv1.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.bn1.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.layer1.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.layer2.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.layer3.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.layer4.parameters(), 'name': 'conv', "lr_mult": 0.1},
    {'params': encoder.fc1.parameters(), 'name': 'ca_cl', "lr_mult": 1.0},
    {'params': encoder.fc2.parameters(), 'name': 'ca_cl', "lr_mult": 1.0},
    ]

    optim_g = optim.SGD(resent_params + discriminator.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    lr_scheduler = StepwiseLR(optim_g, init_lr=args.lr, gamma=0.001, decay_rate=0.75)

    # set criterion
    discriminator.train()
    cls_criterion = nn.CrossEntropyLoss()
    adv_criterion = AdvLoss()
    mse = nn.MSELoss()
    grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    coffee = 1.0
    best_acc,s_acc, t_acc = 0.0, 0.1, 0.0
    g_loss = torch.randn(1)
    cls_loss = torch.randn(1)
    loss = torch.randn(1)
    d_loss = torch.randn(1)
    cls_target = torch.randn(1)

    # set amp trainer
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.max_epoch):
        args.p_adain += (0.5-0.01)*epoch
        encoder.train()
        lr_scheduler.step()
        total_filtered = 0
        filtered_stats = torch.zeros(args.num_classes)
        filtered_true_stats = torch.zeros(args.num_classes)
        for step in range(args.iter_epoch):
            s_out, t_concate_out = next(dset_loaders['source']), next(concate_loaders['train'])

            s_img = s_out['img'].cuda()
            t_concate_img = t_concate_out['img'].cuda()
            s_lab_origin = s_out['target']
            t_concate_lab_origin = t_concate_out['target']
            s_lab = to_onehot(s_lab_origin, args.num_classes).cuda()
            t_concate_lab = to_onehot(t_concate_lab_origin, args.num_classes).cuda()
            s_lab_origin = s_lab_origin.cuda()

            # train the neural networks for domain adaptation
            optim_g.zero_grad()

            if args.amp:
                with autocast():
                    imgs = torch.cat((s_img, t_concate_img), dim=0)
                    f1, f2, fpred = encoder(imgs)
                    sf2, tf2 = f2.chunk(2, dim=0)
                    spred, tpred = fpred.chunk(2, dim=0)

                    # scaling amp loss
                    cls_loss = cls_criterion(spred, s_lab_origin)

                    # Calculate weights
                    t_weights = torch.softmax(tpred, dim=1).detach()
                    entropy = torch.sum(-torch.log(t_weights+1e-16)*t_weights, dim=1)
                    filtered = torch.sum(entropy<args.margin)
                    total_filtered += filtered
                    if filtered > 0:
                        _, ind = torch.max(t_weights[entropy<args.margin,:], dim=1)
                        t_weights[entropy<args.margin,:] = 0
                        t_weights[entropy<args.margin,ind] = 1
                        for kk in ind:
                            filtered_stats[kk] += 1
                        for kk in t_concate_lab_origin[entropy<args.margin]:    
                            filtered_true_stats[kk] += 1

                    ff = grl(torch.cat((sf2, tf2), dim=0))
                    d = discriminator(ff)
                    d_s, d_t = d.chunk(2, dim=0)

                    d_label_s = torch.ones((sf2.size(0), args.num_classes)).to(sf2.device)
                    d_label_t = torch.zeros((tf2.size(0), args.num_classes)).to(tf2.device)
                    sd_loss = bce(d_s, d_label_s)*s_lab*0.5
                    td_loss = bce(d_t, d_label_t)*t_weights*0.5
                    d_loss = torch.mean(torch.sum((sd_loss + td_loss), dim=1))

                    loss = cls_loss + d_loss * coffee
                    scaler.scale(loss).backward()
                    scaler.step(optim_g)
                    scaler.update()

            if (step+1)%10 == 0:
                logger.info("Epoch [{}/{}] Step [{}/{}]: loss={:.3f} d_loss={:.3f} cls_loss={:.3f} s_acc={:.4f} t_acc={:.4f} best={:.4f}"
                            .format(epoch + 1,
                            args.max_epoch,
                            step + 1,
                            args.iter_epoch,
                            loss.item(),
                            -d_loss.item(),
                            cls_loss.item(),
                            s_acc,
                            t_acc,
                            best_acc,
                            )
                )

        if (epoch+1) % args.eval_epoch == 0:
            logger.info('Total filtered:{}'.format(total_filtered))
            logger.info('Filtered class:{}'.format(filtered_stats))
            logger.info('Filtered true:{}'.format(filtered_true_stats))
            s_acc = cal_acc(encoder, dset_loaders_source_test, args, logger)
            t_acc = []
            for t in dset_loaders['target_test'].keys():
                t_acc.append(cal_acc(encoder, dset_loaders['target_test'][t], args, logger))
            t_acc = torch.Tensor(t_acc).mean().item()
            logger.info('t_acc_mean:{}'.format(t_acc))

        if t_acc > best_acc:
            best_acc = t_acc
            delete_model('cada/styflip/{}/{}*'.format(args.dataset, dsets['source'].dataset))
            save_model(encoder, 'cada/styflip/{}'.format(args.dataset), "{}-{}-{}-{:.4f}.pth".format(dsets['source'].dataset, encoder.name, epoch+1, best_acc), logger)
            save_model(discriminator, 'cada/styflip/{}'.format(args.dataset), "{}-{}-{}-{:.4f}.pth".format(dsets['source'].dataset, discriminator.name, epoch+1, best_acc), logger)
            save_model(optim_g, 'cada/styflip/{}'.format(args.dataset), "{}-optim-{}-{:.4f}.pth".format(dsets['source'].dataset, epoch+1, best_acc), logger)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTDA_CADA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--dataset', type=str, default='office-home', choices=['office31', 'office-home', 'domainnet', 'minidomainnet', 'office-home-lds', 'office-home-btlds'])
    parser.add_argument('--source', type=int, default=0, help="source dataset subclass")
    parser.add_argument('--max_epoch', type=int, default=30, help="max epochs")
    parser.add_argument('--eval_epoch', type=int, default=1, help="evaluation epoch")
    parser.add_argument('--iter_epoch', type=int, default=500, help="iterations per epoch")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--feat_dim', type=int, default=256, help="feature space dimension")
    parser.add_argument('--hid_dim', type=int, default=1024, help="feature space dimension")
    parser.add_argument('--bs_limit', type=int, default=96, help="maximum batch size limit due to GPU mem")
    parser.add_argument('--log_step_pre', type=int, default=10, help="Step to visualize loss")
    parser.add_argument('--workers', type=int, default=3, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_gamma', type=float, default=0.001, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay (L2 penalty)')
    parser.add_argument('--nesterov', action='store_false', help='whether to use nesterov SGD')
    parser.add_argument('--resume', action='store_true', help='whether to resume training')
    parser.add_argument('--pretrained', action='store_false', help='whether to use ImageNet pretrained model')
    parser.add_argument('--resume_file', type=str, default=None, help="saved file to resume")
    parser.add_argument('--catal', action='store_true', help='whether to align categorical domain')
    parser.add_argument('--sigma', type=float, default=0.1, help='standard deviation of Gaussian for data augmentation operation of blurring')
    parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--temp', type=float, default=0.05, help='calibration temperature for prototype classifier')
    parser.add_argument('--margin', type=float, default=0.05, help='confidence or marign for filetering the pseudo labels')
    parser.add_argument('--div', type=str, default='kl', help='measure of prediction divergence between one target instance and its perturbed counterpart')
    parser.add_argument('--rand_aug', action='store_true', help='whether to align categorical domain')
    parser.add_argument('--aug_num', type=int, default=1, help="source dataset subclass")
    parser.add_argument('--sub_log', type=str, default='', help="sub log number")
    parser.add_argument('--amp', action='store_true', help="whether to use amp fp16 for training")
    parser.add_argument('--with_permute_adain', action='store_true', help="whether to use amp fp16 for training")
    parser.add_argument('--p_adain', type=float, default=0.01, help='probability to choose the mixed style sample')
    parser.add_argument('--freq', action='store_true', help="whether to fourier augmentation")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    init_random_seed(args.seed)

    main(args)
