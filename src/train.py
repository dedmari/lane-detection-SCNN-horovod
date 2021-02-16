import argparse
import json
import os
import shutil
import time
import pathlib

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
import dataset
from model import SCNN
#from utils.tensorboard import TensorBoard
from utils.transforms import *
from utils.lr_scheduler import PolyLR

# Import packages for distributed computing
import horovod.torch as hvd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiments/exp0")
    parser.add_argument("--use_workers", default=False, type=lambda x: (str(x).lower() == "true"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=48)
    parser.add_argument("--resume", "-r", action="store_true")
    args = parser.parse_args()
    return args
args = parse_args()


#------------ horovod -----------
seed = 3
# Initialize Horovod
hvd.init()
torch.manual_seed(seed)

if torch.cuda.is_available():
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(seed)



# ------------ config ------------
exp_dir = args.exp_dir
while exp_dir[-1]=='/':
    exp_dir = exp_dir[:-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])

device = torch.device(exp_cfg['device'])


batch_size = args.batch_size
val_batch_size = args.val_batch_size
use_workers = args.use_workers
num_workers = args.num_workers

# ------------ model and logs directory ------------
model_logs_dir = exp_cfg['model_logs_dir']
pathlib.Path(model_logs_dir).mkdir(parents=True, exist_ok=True)

exp_name = exp_cfg['exp_name']
#tensorboard = TensorBoard(model_logs_dir)

# ------------ train data ------------
# # CULane mean, std
# mean=(0.3598, 0.3653, 0.3662)
# std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
transform_train = Compose(Resize(resize_shape), Rotation(2), ToTensor(),
                          Normalize(mean=mean, std=std))
dataset_name = exp_cfg['dataset'].pop('dataset_name')
Dataset_Type = getattr(dataset, dataset_name)
train_dataset = Dataset_Type(Dataset_Path[dataset_name], "train", transform_train)

# Using muliple workers (cpu) for loading data
# Horovod: limit # of CPU threads to be used per worker.

kwargs = {}
if use_workers:
    torch.set_num_threads(num_workers)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate, sampler=train_sampler, **kwargs)

# ------------ val data ------------
transform_val_img = Resize(resize_shape)
transform_val_x = Compose(ToTensor(), Normalize(mean=mean, std=std))
transform_val = Compose(transform_val_img, transform_val_x)
val_dataset = Dataset_Type(Dataset_Path[dataset_name], "val", transform_val)

# Horovod: use DistributedSampler to partition the test data.
val_sampler = torch.utils.data.distributed.DistributedSampler(
val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=val_dataset.collate, sampler=val_sampler, **kwargs )


# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# ------------ preparation ------------
net = SCNN(resize_shape, pretrained=True)
lr_scaler = 1
if torch.cuda.is_available():
    net.cuda()
    # Horovod: Scale learning rate as per number of devices
    if hvd.nccl_built():
        lr_scaler = hvd.local_size()

net = torch.nn.DataParallel(net)
lr = exp_cfg['optim']['lr']
momentum = exp_cfg['optim']['momentum']
weight_decay = exp_cfg['optim']['weight_decay']
nesterov = exp_cfg['optim']['nesterov']

# Horovod: scale learning rate by lr_scaler.
optimizer = optim.SGD(net.parameters(), lr=lr * lr_scaler, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
#compression = hvd.Compression.fp16

# Horovod: wrap optimizer with DistributedOptimizer.
gradient_predivide_factor = 1.0
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     #compression=compression,
                                     op=hvd.Average,
                                     gradient_predivide_factor=gradient_predivide_factor)

lr_scheduler = PolyLR(optimizer, 0.9, **exp_cfg['lr_scheduler'])
best_val_loss = 1e6


def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()

    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)

    train_loss = 0
    train_loss_seg = 0
    train_loss_exist = 0
    progressbar = tqdm(range(len(train_loader)))

    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].cuda()
        segLabel = sample['segLabel'].cuda()
        exist = sample['exist'].cuda()

        optimizer.zero_grad()
        seg_pred, exist_pred, loss_seg, loss_exist, loss = net(img, segLabel, exist)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        iter_idx = epoch * len(train_loader) + batch_idx
        train_loss = loss.item()
        train_loss_seg = loss_seg.item()
        train_loss_exist = loss_exist.item()
        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)


    progressbar.close()

    # Horovod: Modify code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if epoch % 2 == 0 and hvd.rank() == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        save_name = os.path.join(model_logs_dir, exp_name + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("------------------------\n")


# Horovod: Combining loss from multiple worker nodes
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def val(epoch):
    global best_val_loss

    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_seg = 0
    val_loss_exist = 0
    progressbar = tqdm(range(len(val_loader)))

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            img = sample['img'].cuda()
            segLabel = sample['segLabel'].cuda()
            exist = sample['exist'].cuda()

            seg_pred, exist_pred, loss_seg, loss_exist, loss = net(img, segLabel, exist)

            val_loss += loss.item()
            val_loss_seg += loss_seg.item()
            val_loss_exist += loss_exist.item()

        progressbar.set_description("batch loss: {:.3f}".format(loss.item()))
        progressbar.update(1)

    # progressbar.close()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    val_loss /= len(val_sampler)
    val_loss_seg /= len(val_sampler)
    val_loss_exist /= len(val_sampler)

    # Horovod: average metric values across workers.
    val_loss = metric_average(val_loss, 'val_loss')
    #val_loss_seg = metric_average(val_loss_seg, 'val_loss_seg')
    #val_loss_exist = metric_average(val_loss_exist, 'val_loss_exist')

    # Horovod: print and log output only on first rank.
    if hvd.rank() == 0:

        print("------------------------\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_name = os.path.join(model_logs_dir, exp_name + '.pth')
            copy_name = os.path.join(model_logs_dir, exp_name + '_best.pth')
            shutil.copyfile(save_name, copy_name)


def main():

    global best_val_loss
    if args.resume:
        save_dict = torch.load(os.path.join(model_logs_dir, exp_name + '.pth'))
        #if isinstance(net, torch.nn.DataParallel):
        #    net.module.load_state_dict(save_dict['net'])
        #else:
        net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1e6)
    else:
        start_epoch = 0

    # exp_cfg['MAX_EPOCHES'] = int(np.ceil(exp_cfg['lr_scheduler']['max_iter'] / len(train_loader)))
    for epoch in range(start_epoch, exp_cfg['MAX_EPOCHES']):
        train(epoch)
        if epoch % 10 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()
