import socket
import timeit
from datetime import datetime
import cv2 as cv
import os
import glob
from collections import OrderedDict
import numpy as np

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import scipy.misc as m
from PIL import Image

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import cityscapes
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from dataloaders import custom_transforms as tr
gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img -= self.mean
        img /= 255.0
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img
# Network definition
net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=19, os=16, pretrained=True)

net.load_state_dict(torch.load('./run/run_7/models/deeplabv3plus-xception-cityscapes_epoch-600.pth'))
#net.load_state_dict(torch.load('./run/run_8/models/deeplabv3plus-xception-cityscapes_epoch-129.pth'))

criterion = utils.cross_entropy2d

transform_img = transforms.Compose([
    Normalize_cityscapes(mean=(72.39, 82.91, 73.16)),
    ToTensor()])


if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

kitti_train_path = '/habtegebrialdata/Datasets/KittiSceneFlow/testing/image_2'
kitti_test_path = '/habtegebrialdata/Datasets/KittiSceneFlow/testing/image_3'

left_path = '/habtegebrialdata/Datasets/KittiSceneFlow/deeplab/training/image_2'
right_path = '/habtegebrialdata/Datasets/KittiSceneFlow/deeplab/training/image_3'

train_files = sorted(recursive_glob(kitti_train_path, '.png'))
test_files = sorted(recursive_glob(kitti_test_path, '.png'))
output_dir = 'deeplab_v3_on_kitti_stereo'
log_dir = os.path.join(output_dir, 'run')
writer = SummaryWriter(log_dir=log_dir)
cnt = 0
net.eval()

for tr_path, ts_path in zip(train_files, test_files):
    print(cnt)
    _train = Image.open(tr_path).convert('RGB')
    train_im = transform_img(_train)
    _test = Image.open(ts_path).convert('RGB')
    test_im = transform_img(_test)
    inp_img = torch.stack([train_im, test_im]).cuda(gpu_id)
    with torch.no_grad():
        outputs = net.forward(inp_img)

    preds = torch.max(outputs, 1)[1].cpu()
    b, h, w = preds.shape
    preds = preds.unsqueeze(3).expand(b, h, w, 3).numpy()
    left_name = tr_path.replace(kitti_train_path, left_path)
    right_name = tr_path.replace(kitti_test_path, right_path)
    os.makedirs(os.path.dirname(left_name), exist_ok=True)
    os.makedirs(os.path.dirname(right_name), exist_ok=True)
    cv.imwrite(left_name, preds[0])
    cv.imwrite(right_name, preds[1])
    grid_image = make_grid(
                utils.decode_seg_map_sequence(torch.max(outputs, 1)[1].detach().cpu().numpy(), 'cityscapes'), 3,
                normalize=False, range=(0, 255)
                )
    writer.add_image('PredictedLabel', grid_image, cnt)
    grid_image = make_grid(inp_img.clone().cpu().data, 3, normalize=True)
    writer.add_image('Image', grid_image, cnt)
    cnt += 1
    #exit()


"""
if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    transform_img = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomScale((0.5, 0.75)),
        tr.RandomCrop((512, 1024)),
        tr.RandomRotate(5),
        tr.Normalize_cityscapes(mean=(72.39, 82.91, 73.16)),
        tr.ToTensor()])

    cityscapes_train = cityscapes.CityscapesSegmentation(split='train',
                                                         transform=transform_img)

    trainloader = DataLoader(cityscapes_train, batch_size=1, shuffle=False, num_workers=0)
    testloader = DataLoader(cityscapes_test, batch_size=1, shuffle=False, num_workers=0)

    utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_vl = 0.0
    running_loss_ts = 0.0
    previous_miou = -1.0
    aveGrad = 0
    global_step = 0
    print("Training Network")
    images =
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            if ii % (num_img_tr // 10) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy(), 'cityscapes'), 3,
                    normalize=False,
                    range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(
                    utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy(), 'cityscapes'), 3,
                    normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
    writer.close()
"""
