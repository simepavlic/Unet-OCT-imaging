import sys
import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from data.fetcher import DatasetFetcher
from nn.unet_model import UNet

from PIL import Image


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def train_net(net,
            fetcher,
              criterion,
              epochs=5,
              batch_size=1,
              lr=0.05,
              gpu=False):

    train_files = fetcher.get_train_files()

    print('''
        Starting training:
            Epochs: {}
            Batch size: {}
            Learning rate: {}
            Training size: 80%
            Validation size: 20%
            CUDA: {}
        '''.format(epochs, batch_size, lr,
                   str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        train_origs = [fetcher.get_image_matrix(i) for i in train_files[0]]
        train_masks = [fetcher.get_mask_matrix(i) for i in train_files[1]]
        train = zip(train_origs, train_masks)

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b])
            masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            masks = torch.from_numpy(masks)

            imgs = imgs.view(imgs.size()[0], 1, imgs.size()[1], imgs.size()[2])

            if gpu:
                imgs = imgs.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()

            outputs = net(imgs)
            outputs_t = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 4)

            masks_t = masks.view(-1)

            loss = criterion(outputs_t, masks_t)

            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.item()))

            loss.backward()
            optimizer.step()

    project_root = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(project_root + '/../model.pth')
    torch.save(net.state_dict(), dest)
    print('Finished Training')


def validate_net(net, fetcher, criterion, gpu=False):
    val_files = fetcher.get_valid_files()

    val_origs = [fetcher.get_image_matrix(i) for i in val_files[0]]
    val_masks = [fetcher.get_mask_matrix(i) for i in val_files[1]]
    val = zip(val_origs, val_masks)

    net.eval()

    total_loss = 0

    for i, b in enumerate(val):
        img = b[0]
        mask = b[1]

        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)

        img = img.view(1, 1, img.size()[0], img.size()[1])

        if gpu:
            img = img.cuda()
            mask = mask.cuda()

        output = net(img)

        output_t = output.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        mask_t = mask.view(-1)

        loss = criterion(output_t, mask_t)

        total_loss += loss.item()
    print('Average validation loss: %f' % (total_loss / len(val_origs)))


def test_net(net, fetcher, output_path, gpu=False):

    test_files = fetcher.get_test_files()
    test_imgs = [fetcher.get_image_matrix(i) for i in test_files]

    net.eval()

    for b, file_path in zip(test_imgs, test_files):
        img = torch.from_numpy(b)
        img = img.view((1, 1, img.size()[0], img.size()[1]))

        if gpu:
            img = img.cuda()

        output = net(img)

        predicted = torch.argmax(output, dim=1)

        predicted = predicted.view(predicted.size()[1], predicted.size()[2])
        predicted = predicted.cpu()
        predicted = predicted.numpy()
        predicted[predicted == 1] = 80
        predicted[predicted == 2] = 160
        predicted[predicted == 3] = 240
        img = Image.fromarray(predicted.astype('uint8'), 'L')
        file = os.path.basename(file_path)
        img_dest = os.path.join(output_path, file)
        img.save(img_dest)


if __name__ == '__main__':

    net = UNet(n_channels=1, n_classes=4)
    fetcher = DatasetFetcher()
    fetcher.fetch_dataset()
    gpu = True
    criterion = nn.CrossEntropyLoss()

    if gpu:
        net.cuda()

    try:
        train_net(net=net,
                  fetcher=fetcher,
                  criterion=criterion,
                  epochs=10,
                  batch_size=1,
                  lr=0.05,
                  gpu=gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    validate_net(net, fetcher, criterion, gpu=gpu)

    project_root = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(project_root + '/../output')

    test_net(net, fetcher, dest, gpu=gpu)
