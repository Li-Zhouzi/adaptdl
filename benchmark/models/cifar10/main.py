'''Train CIFAR10 with PyTorch.'''
import time
from datetime import datetime

# Log Python script start time
print(f"[TIMING] Python main.py started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} ({time.time()})", flush=True)
# Timing helpers for pre-training steps
_t0 = time.perf_counter()
_step_records = []

def _step_begin(name):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[TIMING] {name} started at: {ts} ({time.time():.3f})", flush=True)
    return time.perf_counter()

def _step_end(name, start):
    end = time.perf_counter()
    duration = end - start
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    total = end - _t0
    print(f"[TIMING] {name} completed at: {ts} ({time.time():.3f}) | duration={duration:.3f}s | total={total:.3f}s", flush=True)
    _step_records.append((name, ts, duration, total))

def _print_pretraining_summary():
    print("[TIMING] Pre-training steps summary:", flush=True)
    for idx, (name, ts, duration, total) in enumerate(_step_records, 1):
        print(f"[TIMING]   {idx}. {name}: duration={duration:.3f}s, completed at {ts}, total_since_start={total:.3f}s", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import adaptdl
import adaptdl.torch

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--epochs', default=40, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

_s = _step_begin("Load training dataset")
trainset = torchvision.datasets.CIFAR10(root="/mnt", train=True, download=False, transform=transform_train)
_step_end("Load training dataset", _s)
print("trainset length:", len(trainset))
_s = _step_begin("Create AdaptiveDataLoader (train)")
trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024),
                                 gradient_accumulation=True)
_step_end("Create AdaptiveDataLoader (train)", _s)

_s = _step_begin("Load validation dataset")
validset = torchvision.datasets.CIFAR10(root="/mnt", train=False, download=False, transform=transform_test)
_step_end("Load validation dataset", _s)
validloader = adaptdl.torch.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
_s = _step_begin("Build model")
net = eval(args.model)()
_step_end("Build model", _s)
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
_s = _step_begin("Move model to device")
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True
_step_end("Move model to device", _s)

_s = _step_begin("Create optimizer and scheduler")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))
_step_end("Create optimizer and scheduler", _s)

_s = _step_begin("Log environment variables")
print(f"[DEBUG] Environment variables at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}", flush=True)
print(f"[DEBUG] ADAPTDL_MASTER_ADDR: {os.getenv('ADAPTDL_MASTER_ADDR', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_MASTER_PORT: {os.getenv('ADAPTDL_MASTER_PORT', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_SUPERVISOR_URL: {os.getenv('ADAPTDL_SUPERVISOR_URL', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_REPLICA_RANK: {os.getenv('ADAPTDL_REPLICA_RANK', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_NUM_REPLICAS: {os.getenv('ADAPTDL_NUM_REPLICAS', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_JOB_ID: {os.getenv('ADAPTDL_JOB_ID', 'NOT SET')}", flush=True)
print(f"[DEBUG] ADAPTDL_NUM_RESTARTS: {os.getenv('ADAPTDL_NUM_RESTARTS', 'NOT SET')}", flush=True)
_step_end("Log environment variables", _s)

_s = _step_begin("init_process_group(nccl)")
adaptdl.torch.init_process_group("nccl")
_step_end("init_process_group(nccl)", _s)

_s = _step_begin("Create AdaptiveDataParallel")
net = adaptdl.torch.AdaptiveDataParallel(net, optimizer, lr_scheduler)
_step_end("Create AdaptiveDataParallel", _s)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    stats = adaptdl.torch.Accumulator()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item()

        trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")

    with stats.synchronized():
        total = stats.get("total", 0)
        if total:
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            stats["accuracy"] = stats["correct"] / stats["total"]
            writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
            writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
            report_train_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
            print("Train:", stats)
        else:
            # No batches processed this epoch window; skip logging to avoid KeyError.
            print("Train: skipped (no batches processed)")

def valid(epoch):
    net.eval()
    stats = adaptdl.torch.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()
            print("HERE", stats["loss_sum"])

    with stats.synchronized():
        total = stats.get("total", 0)
        if total:
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            stats["accuracy"] = stats["correct"] / stats["total"]
            writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
            writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
            report_valid_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
            print("Valid:", stats)
        else:
            # No batches processed this epoch window; skip logging to avoid KeyError.
            print("Valid: skipped (no batches processed)")

_print_pretraining_summary()
print(f"[TIMING] Initialization complete, starting training loop at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} ({time.time()})", flush=True)

with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
        train(epoch)
        valid(epoch)
        lr_scheduler.step()