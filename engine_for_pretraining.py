import math
import sys
from typing import Iterable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import mae_utils
from einops import rearrange
import copy
from frft_torch import frft, ifrft
import numpy as np


class frftLoss(nn.Module):
    def __init__(self):
        super(frftLoss, self).__init__()

    def forward(self, input, target):
        tmp = (input - target) ** 2
        if input.shape[-1] == 2:
            loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12)
        else:
            loss = torch.sqrt(tmp + 1e-12)
        return loss


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, patch_size: tuple,
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None):
    model.train()
    metric_logger = mae_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', mae_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', mae_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)

    loss_func = frftLoss()#nn.MSELoss()

    for step, (ori_images, images, bool_masked_pos) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it]
        images = images.to(device, non_blocking=True)
        ori_images = ori_images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            if normlize_target:
                images_squeeze = rearrange(ori_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(ori_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1])

            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)
            ##########################
            rec_labels = rearrange(labels, 'b n (p c) -> b c n p', c=1)
            labels_list = []
            order_array = np.arange(0, 4.1, 1)
            for i in range(order_array.shape[0]):
                frac_list = []
                for j in range(labels.shape[0]):
                    frac_ = frft(rec_labels[j], order_array[i]).unsqueeze(0)
                    frac_list.append(frac_)
                frac = torch.cat(frac_list, dim=0)
                labels_list.append(frac)
        outputs = model(images, bool_masked_pos)
        rec_outputs = rearrange(outputs, 'b n (p c) -> b c n p', c=1)
        outputs_list = []
        order_array = np.arange(0, 4.1, 1)
        for i in range(order_array.shape[0]):
            frac_list = []
            for j in range(outputs.shape[0]):
                frac_ = frft(rec_outputs[j], order_array[i]).unsqueeze(0)
                frac_list.append(frac_)
            frac = torch.cat(frac_list,dim=0)
            outputs_list.append(frac)
        loss = 0
        for i in range(len(outputs_list)):
            loss += torch.mean(loss_func(input=outputs_list[i], target=labels_list[i])).type(torch.float)
        ######################
        # outputs = model(images, bool_masked_pos)
        # loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        metric_logger.update(train_loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:
            log_writer.update(train_loss=loss_value, head="train_loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    print("Train Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, epoch: int, patch_size: tuple,
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None):
    model.eval()
    metric_logger = mae_utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader)
    loss_func = frftLoss()#nn.MSELoss()
    loss_average = 0.
    for step, (ori_images, images, bool_masked_pos) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        ori_images = ori_images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        with torch.no_grad():
            if normlize_target:
                images_squeeze = rearrange(ori_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(ori_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1])
            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)
            ##############################
            rec_labels = rearrange(labels, 'b n (p c) -> b c n p', c=1)
            labels_list = []
            order_array = np.arange(0, 4.1, 1)
            for i in range(order_array.shape[0]):
                frac_list = []
                for j in range(labels.shape[0]):
                    frac_ = frft(rec_labels[j], order_array[i]).unsqueeze(0)
                    frac_list.append(frac_)
                frac = torch.cat(frac_list, dim=0)
                labels_list.append(frac)
        outputs = model(images, bool_masked_pos)
        rec_outputs = rearrange(outputs, 'b n (p c) -> b c n p', c=1)
        outputs_list = []
        order_array = np.arange(0, 4.1, 1)
        for i in range(order_array.shape[0]):
            frac_list = []
            for j in range(outputs.shape[0]):
                frac_ = frft(rec_outputs[j], order_array[i]).unsqueeze(0)
                frac_list.append(frac_)
            frac = torch.cat(frac_list, dim=0)
            outputs_list.append(frac)
        loss = 0.
        for i in range(len(outputs_list)):
            loss += torch.mean(loss_func(input=outputs_list[i], target=labels_list[i])).type(torch.float)
        ###########################################
        # outputs = model(images, bool_masked_pos)
        # loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()
        loss_average += loss_value
        metric_logger.update(valid_loss=loss_value)
        if log_writer is not None:
            log_writer.update(valid_loss=loss_value, head="valid_loss")
            log_writer.set_step()
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    print("Valid Averaged stats:", metric_logger)
    return loss_average/(start_steps+1)


def gen_data_visualization(model: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, patch_size: tuple, window_size: tuple,
                    normlize_target: bool = True):
    model.eval()
    metric_logger = mae_utils.MetricLogger(delimiter="  ")
    print_freq = len(data_loader)
    os.makedirs('./gen_result',exist_ok=True)
    loss_func = nn.MSELoss()
    for step, (images, bool_masked_pos) in enumerate(metric_logger.log_every(data_loader, print_freq)):
        images = images.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        with torch.no_grad():
            if normlize_target:
                images_squeeze = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[1])
                images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
            else:
                images_patch = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1])
            B, _, C = images_patch.shape
            labels = images_patch[bool_masked_pos].reshape(B, -1, C)
        outputs = model(images, bool_masked_pos)
        loss = loss_func(input=outputs, target=labels)
        loss_value = loss.item()
        print(loss_value/labels.shape[0])

        # original data
        ori_image = copy.deepcopy(images)

        # mask data
        mask = torch.ones_like(images_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=1)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=window_size[0], w=window_size[1])

        # reconstruction data
        B, _, C = images_patch.shape
        images_patch[bool_masked_pos] = outputs.reshape(-1,250)
        images_patch = images_patch.reshape(B, -1, C)
        rec_img = rearrange(images_patch, 'b n (p c) -> b n p c', c=1)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=window_size[0], w=window_size[1])


        for batch_ind in range(images.shape[0]):
            plt.figure()
            for channel_ind in range(5):
                plt.subplot(5, 1, channel_ind+1)
                plt.plot(torch.squeeze(rec_img[batch_ind,:,channel_ind,:]).detach().numpy(),'b')
                plt.plot(torch.squeeze(ori_image[batch_ind,:,channel_ind,:]).detach().numpy(),'r')
                plt.plot(torch.squeeze(mask[batch_ind,:,channel_ind,:]).detach().numpy(),'k')
            plt.savefig('./gen_result/{}_{}.pdf'.format(step,batch_ind))
            plt.show()