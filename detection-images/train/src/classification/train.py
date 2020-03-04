import os
import time
import datetime
import sys

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
import utils
from datasets import TrackClassification as Dataset
from models import ResNet as Model


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg




    



#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     model = model.to(device)
#     if torch.cuda.device_count() > 1:
#         model = nn.DataParallel(model)
        
#     best_acc = 0
#     best_state = model.state_dict()

#     train_losses = []
#     train_accs = []
#     val_losses = []
#     val_accs = []
#     for epoch in range(epochs):
        
#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         # training
#         train_loss = 0
#         train_acc = 0
#         for i, (inputs, labels) in enumerate(data['train']):

#             labels -= 1
            
#             model.train()            
            

#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = cross_entropy(outputs, labels)
#             acc = accuracy(outputs, labels)[0]

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * inputs.size(0)
#             train_acc += acc.item() * inputs.size(0)

#         train_loss = train_loss / len(data['train'].dataset)
#         train_acc = train_acc / len(data['train'].dataset)


#         print(
#             f'[info] epoch: {epoch:>3d}'
#             f' - train_loss: {train_loss:.4f} - train_acc: {train_acc:6.2%}'
#             f' - time: {str(datetime.now().time()).split(".")[0]}'
#         )

#         train_losses.append(train_loss)
#         train_accs.append(train_acc)

#     result = {
#         'train_loss': train_losses,
#         'train_acc': train_accs,
#     }
    
#     return model, result


if __name__ == "__main__":
    
    output_dir = 'models/cls/'
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = 20
    
    train_set = Dataset(os.path.realpath('data/images4cls/train/'), train=True)
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)
    
    valid_set = Dataset(os.path.realpath('data/images4cls/valid/'), train=True)
    valid_loader = DataLoader(valid_set, batch_size=20, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = Model()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4)
#     optimizer = torch.optim.SGD(
#         params=model.parameters(), 
#         lr=0.1, 
#         momentum=0.9, 
#         weight_decay=5e-4,
#         nesterov=True
#     )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, 
        milestones=[6, 12, 16],
        gamma=0.2
    )
    
    print("Start training")
    start_time = time.time()
    for epoch in range(epochs):
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        evaluate(model, criterion, valid_loader, device=device)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(
            checkpoint,
            os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
        torch.save(
            checkpoint,
            os.path.join(output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))