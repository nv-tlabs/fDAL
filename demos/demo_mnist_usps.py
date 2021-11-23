# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# The MIT License (MIT)
# Copyright (c) 2020 JunguangJiang
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Performance 95.7  at 22 Epochs.
from tqdm import tqdm
from fDAL import fDALLearner
from torchvision import transforms
import torch.nn
import torch.nn as nn
from torch.utils.data import DataLoader
from data_list import ImageList, ForeverDataIterator
import torch.optim as optim
import numpy as np
import random
import fire
import os
import torch.nn.utils.spectral_norm as sn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


def build_network():
    # network encoder...
    lenet = nn.Sequential(
        nn.Conv2d(1, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(20, 50, kernel_size=5),
        nn.Dropout2d(p=0.5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
    )

    # create a bootleneck layer. it usually helps
    bottleneck_dim = 500
    bottleneck = nn.Sequential(
        nn.Linear(800, bottleneck_dim),
        nn.BatchNorm1d(bottleneck_dim),
        nn.LeakyReLU(),
        nn.Dropout(0.5)
    )

    backbone = nn.Sequential(
        lenet,
        bottleneck
    )

    # classification head
    num_classes = 10
    taskhead = nn.Sequential(
        sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        sn(nn.Linear(bottleneck_dim, num_classes)),
    )

    return backbone, taskhead, num_classes


def build_data_loaders():
    source_list = './data_demo/usps2mnist/mnist_train.txt'
    target_list = './data_demo/usps2mnist/usps_train.txt'
    test_list = './data_demo/usps2mnist/usps_test.txt'
    batch_size = 128

    # training loaders....
    train_source = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    train_target = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]), mode='L'),
        batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    return train_source, train_target, test_loader


def scheduler(optimizer_, init_lr_, decay_step_, gamma_):
    class DecayLRAfter:
        def __init__(self, optimizer, init_lr, decay_step, gamma):
            self.init_lr = init_lr
            self.gamma = gamma
            self.optimizer = optimizer
            self.iter_num = 0
            self.decay_step = decay_step

        def get_lr(self) -> float:
            if ((self.iter_num + 1) % self.decay_step) == 0:
                lr = self.init_lr * self.gamma
                self.init_lr = lr

            return self.init_lr

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

    return DecayLRAfter(optimizer_, init_lr_, decay_step_, gamma_)


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


def prepare_data_if_first_time():
    if os.path.exists('./data_demo/usps2mnist/mnist_train_image') is False:
        print('unpacking demo data...')
        fname = 'images.tar' if os.path.isfile('./data_demo/usps2mnist/images.tar') else 'images.tar.gz'
        if os.system(f'cd ./data_demo/usps2mnist/ && tar -xf {fname} && echo "done"') != 0:
            print(f'{fname} not found in ./data_demo/usps2mnist/. Check that you downloaded the dataset.')
            return False
    return True


def sample_batch(train_source, train_target, device):
    x_s, labels_s = next(train_source)
    x_t, _ = next(train_target)
    x_s = x_s.to(device)
    x_t = x_t.to(device)
    labels_s = labels_s.to(device)
    return x_s, x_t, labels_s


def main(divergence='pearson', n_epochs=30, iter_per_epoch=3000, lr=0.01, wd=0.002, reg_coef=0.5, seed=2):
    seed_all(seed)

    # unzip datasets if this is first run.
    if prepare_data_if_first_time() is False:
        return False

    # build the network.
    backbone, taskhead, num_classes = build_network()

    # build the dataloaders.
    train_source, train_target, test_loader = build_data_loaders()

    # define the loss function....
    taskloss = nn.CrossEntropyLoss()

    # fDAL ----
    train_target = ForeverDataIterator(train_target)
    train_source = ForeverDataIterator(train_source)
    learner = fDALLearner(backbone, taskhead, taskloss, divergence=divergence, reg_coef=reg_coef, n_classes=num_classes,
                          grl_params={"max_iters": 3000, "hi": 0.6, "auto_step": True}  # ignore for defaults.
                          )
    # end fDAL---

    #
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    learner = learner.to(device)

    # define the optimizer.

    # Hyperparams and scheduler follows CDAN.
    opt = optim.SGD(learner.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    opt_schedule = scheduler(opt, lr, decay_step_=iter_per_epoch * 5, gamma_=0.5)

    print('Starting training...')
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
            # avoid gradient issues if any early on training.
            torch.nn.utils.clip_grad_norm_(learner.parameters(), 10)
            opt.step()
            if i % 1500 == 0:
                print(f"Epoch:{epochs} Iter:{i}. Task Loss:{others['taskloss']}")

        test_acc, test_loss = test_accuracy(learner.get_reusable_model(True), test_loader, taskloss, device)
        print(f"Epoch:{epochs} Test Acc: {test_acc} Test Loss: {test_loss}")

    # save the model.
    torch.save(learner.get_reusable_model(True).state_dict(), './checkpoint.pt')
    print('done.')


if __name__ == "__main__":
    fire.Fire(main)
