from tqdm import tqdm
import torch
import torch.nn as nn

def cal_acc(encoder, test_set, args, logger=None):
    encoder.eval()
    loss, acc = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for image_folder in tqdm(test_set):
            images = image_folder['img']
            labels = image_folder['target']
            images = images.cuda()
            labels = labels.cuda()
            if encoder.name == 'ResNet-IRM':
                _,_,preds,preds_env = encoder(images)
            else:
                _,_,preds = encoder(images)
            loss += criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(test_set)
    acc /= len(test_set.dataset)
    if logger == None:
        print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    else:
        logger.info("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

    return acc

def cal_accuracy(encoder, test_set, args):
    '''
    calculate the accuracy of each calss
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    encoder.eval()
    total_vector = torch.FloatTensor(args.num_classes).fill_(0)
    correct_vector = torch.FloatTensor(args.num_classes).fill_(0)
    for image_folder in tqdm(test_set):
        images = image_folder['img']
        labels = image_folder['target']
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            _,_,preds = encoder(images)
            loss = criterion(preds, labels)

            prec1, prec5 = accuracy(preds.data, labels, topk=(1, 5))
            total_vector, correct_vector = accuracy_for_each_class(preds.data, labels, total_vector, correct_vector) # compute class-wise accuracy
            losses.update(loss.data.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

    acc_for_each_class = 100.0 * correct_vector / total_vector
    print(' * Test on T test set - Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print('\nAcc for each class: ')
    for i in range(args.num_classes):
        print("%d: %3f" % (i+1, acc_for_each_class[i]))


    