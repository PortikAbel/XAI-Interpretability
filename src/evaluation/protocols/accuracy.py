from tqdm import tqdm
from enum import Enum
import torch
from torch.utils.data import DataLoader

from data.funny_birds import FunnyBirds


def accuracy_protocol(model, args):
    class Summary(Enum):
        NONE = 0
        AVERAGE = 1
        SUM = 2
        COUNT = 3

    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
            self.name = name
            self.fmt = fmt
            self.summary_type = summary_type
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

        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)

        def summary(self):
            fmtstr = ""
            if self.summary_type is Summary.NONE:
                fmtstr = ""
            elif self.summary_type is Summary.AVERAGE:
                fmtstr = "{name} {avg:.3f}"
            elif self.summary_type is Summary.SUM:
                fmtstr = "{name} {sum:.3f}"
            elif self.summary_type is Summary.COUNT:
                fmtstr = "{name} {count:.3f}"
            else:
                raise ValueError("invalid summary type %r" % self.summary_type)

            return fmtstr.format(**self.__dict__)

    def accuracy(output, target, topk=(1,)):
        """Computes accuracy over k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    transforms = None

    test_dataset = FunnyBirds(args.data, "test", transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    for samples in tqdm(test_loader):
        images = samples["image"]
        target = samples["target"]
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

    print(top1)
    print(top5)

    return top1.avg.item() / 100
