# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tqdm import tqdm
from fDAL import fDALLearner
from torchvision import transforms
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
from loaders.data_list import DAImageList, ForeverDataIterator
import torch.optim as optim
import numpy as np
import random
import fire
import os
import torch.nn.utils.spectral_norm as sn
from collections import OrderedDict
from resnet import resnet50


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


def build_network(pretrained=True, bottleneck_dim=1024):
    # network encoder...
    resnet = resnet50(pretrained=pretrained)

    # create a bootleneck layer as in... MDD.
    bottleneck = nn.Sequential(
        nn.Linear(resnet.out_features, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.ReLU(),
        nn.Dropout(0.5)
    )

    backbone = nn.Sequential(
        OrderedDict([
            ('resnet', resnet),
            ('bottleneck', bottleneck)
        ])
    )

    num_classes = 31

    # classification head
    taskhead = nn.Sequential(
        sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        sn(nn.Linear(bottleneck_dim, num_classes))
    )

    # Initialization follows from MDD
    bottleneck[0].weight.data.normal_(0, 0.005)
    bottleneck[0].bias.data.fill_(0.1)

    taskhead[0].weight.data.normal_(0, 0.01)
    taskhead[0].bias.data.fill_(0.0)

    taskhead[-1].weight.data.normal_(0, 0.01)
    taskhead[-1].bias.data.fill_(0.0)

    return backbone, taskhead, num_classes


def build_data_loaders(root_src, source, target, workers, batch_size, num_classes=31):
    ImageNetNormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    class ResizeImage(object):
        """Resize the input PIL Image to the given size.

        Args:
            size (sequence or int): Desired output size. If size is a sequence like
                (h, w), output size will be matched to this. If size is an int,
                output size will be (size, size)
        """

        def __init__(self, size):
            if isinstance(size, int):
                self.size = (int(size), int(size))
            else:
                self.size = size

        def __call__(self, img):
            th, tw = self.size
            return img.resize((th, tw))

    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ImageNetNormalize
    ])

    # Test Transform....
    val_transform = transforms.Compose([
        ResizeImage(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ImageNetNormalize
    ])

    cls_name = [f"{i}" for i in range(num_classes)]
    train_source = DataLoader(DAImageList(root_src, cls_name, source, transform=train_transform),
                              batch_size=batch_size,
                              shuffle=True, num_workers=workers, drop_last=True, pin_memory=True
                              )
    train_target = DataLoader(DAImageList(root_src, cls_name, target, transform=train_transform),
                              batch_size=batch_size,
                              shuffle=True, num_workers=workers, drop_last=True, pin_memory=True
                              )

    # validation target data....
    val_loader = DataLoader(DAImageList(root_src, [i for i in range(num_classes)], target, transform=val_transform),
                            batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True)

    return train_source, train_target, val_loader


def scheduler(optimizer_, init_lr_, decay_step_, gamma_):
    # The MIT License (MIT)
    # Copyright (c) 2020 JunguangJiang
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    class INVScheduler:
        """
        LR scheduler and hyperparameters obtained from MDD.
        source: https://github.com/thuml/Transfer-Learning-Library/blob/97f67c60095956b0d2206d48d729e4b51d39192a/tools/lr_scheduler.py

        This learning rate scheduler updates the learning rate as:

        .. math::
            \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},


        where `i` is the iteration steps.

        Parameters:
            - **optimizer**: Optimizer
            - **init_lr** (float, optional): initial learning rate. Default: 0.01
            - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
            - **decay_rate** (float, optional): :math:`p` . Default: 0.75
        """

        def __init__(self, optimizer, init_lr=0.01, gamma=0.001, decay_rate=0.75):
            self.init_lr = init_lr
            self.gamma = gamma
            self.decay_rate = decay_rate
            self.optimizer = optimizer
            self.iter_num = 0

        def get_lr(self):
            lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
            return lr

        def step(self):
            """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

            self.iter_num += 1

        def __str__(self):
            return str(self.__dict__)

    return INVScheduler(optimizer_, init_lr_, gamma_, decay_step_)


def test_accuracy(model, loader, loss_fn, device):
    avg_acc = 0.
    avg_loss = 0.
    n = len(loader.dataset)
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            yhat = model(x)
            avg_loss += (loss_fn(yhat, y).item() / n)

            pred = yhat.max(1, keepdim=True)[1]
            avg_acc += (pred.eq(y.view_as(pred)).sum().item() / n)

    return avg_acc, avg_loss


def sample_batch(train_source, train_target, device):
    x_s, labels_s = next(train_source)
    x_t, _ = next(train_target)
    x_s = x_s.to(device)
    x_t = x_t.to(device)
    labels_s = labels_s.to(device)
    return x_s, x_t, labels_s


def main(divergence='pearson',
         root_src='./data/office31/', source='image_list/amazon.txt', target='image_list/webcam.txt',
         batch_size=32, n_epochs=40, iter_per_epoch=2000,
         lr=0.004, wd=0.0009, reg_coef=1.0, bottleneck_dim=1024,
         workers=8, seed=2):
    seed_all(seed)

    # build the network.
    backbone, taskhead, num_classes = build_network(bottleneck_dim=bottleneck_dim)

    # build the dataloaders.
    train_source, train_target, test_loader = build_data_loaders(root_src=root_src,
                                                                 source=os.path.join(root_src, source),
                                                                 target=os.path.join(root_src, target), workers=workers,
                                                                 batch_size=batch_size)

    # define the loss function....
    taskloss = nn.CrossEntropyLoss()

    # fDAL ----
    train_target = ForeverDataIterator(train_target)
    train_source = ForeverDataIterator(train_source)

    learner = fDALLearner(backbone, taskhead, taskloss, divergence=divergence, reg_coef=reg_coef, n_classes=num_classes,
                          grl_params={"max_iters": 1000, "hi": 0.1, "auto_step": True}  # ignore for defaults.
                          )
    # end fDAL---

    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learner = learner.to(device)

    # define the optimizer.

    # If we do not need to specify lr_mult, just do
    # opt = optim.SGD(learner.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

    opt = optim.SGD(
        [{"params": learner.backbone.resnet.parameters(), "lr_mult": 0.1},
         {"params": learner.backbone.bottleneck.parameters(), "lr_mult": 1.0},
         {"params": learner.taskhead.parameters(), "lr_mult": 1.0},
         {"params": learner.auxhead.parameters(), "lr_mult": 1.0}],
        lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)

    opt_schedule = scheduler(opt, lr, decay_step_=0.75, gamma_=0.0002)
    best_acc = -np.inf

    print('+ Starting training...')
    for epochs in range(n_epochs):
        learner.train()
        for i in range(iter_per_epoch):
            opt_schedule.step()
            # batch data loading...
            x_s, x_t, labels_s = sample_batch(train_source, train_target, device)
            # forward and loss
            loss, others = learner((x_s, x_t), labels_s)
            # opt stuff
            opt.zero_grad()
            loss.backward()

            # avoid gradient issues (if any) early on training with some divergences.
            # Not need it if spectral_norm is used in the classifiers h and h'.
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10)
            opt.step()

            if i % 100 == 0:
                print(
                    f"Epoch:{epochs} Iter:{i}. Task Loss:{others['taskloss']:.4f} | \t f-DAL Src: {others['fdal_src']:.4f} \t f-DAL Trg: {others['fdal_trg']:.4f}")

        test_acc, test_loss = test_accuracy(learner.get_reusable_model(True), test_loader, taskloss, device)
        best_acc = max(test_acc, best_acc)
        print(f"Epoch:{epochs} Test Acc: {test_acc} Test Loss: {test_loss}. Best Acc: {best_acc}")

    # save the model.
    torch.save(learner.get_reusable_model(True).state_dict(), './checkpoint.pt')
    print(f'best_acc:{best_acc}')
    print('done.')


if __name__ == "__main__":
    # example python demos/demo_office31.py --divergence=pearson --root_src=directory_with_office_31_dataset (e.g. /data/office31/)
    fire.Fire(main)
