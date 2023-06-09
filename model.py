'''
author:new-star
time:8.4.2023
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["pointCoder", "pointwhCoder"]


class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1.), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        # self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_y, patch_stride_x = 1. / self.patch_count[0], 1. / self.patch_count[1]
        for i in range(self.patch_count[0]):
            for j in range(self.patch_count[1]):
                y = (0.5 + i) * patch_stride_y
                x = (0.5 + j) * patch_stride_x
                anchors.append([x, y])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        # self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1. / self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel if self.tanh else rel_codes[:, :, 0] * pixel / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel if self.tanh else rel_codes[:, :, 1] * pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=0., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size


class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1.), pts=1, tanh=True, wh_bias=None):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes, img_size, output_size):  # 对预测的偏移量归一化到合适的值，类似于将其编码
        self.input_size = img_size
        self.patch_count = output_size
        self._generate_anchor(device=boxes.device)
        if self.wh_bias is not None:
            boxes[:, :, 2:] = boxes[:, :, 2:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x, pixel_y = 1. / self.patch_count[1], 1. / self.patch_count[0]
        wx, wy, wh, ww = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel_x if self.tanh else rel_codes[:, :, 0] * pixel_x / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel_y if self.tanh else rel_codes[:, :, 1] * pixel_y / wy

        dw = F.relu(F.tanh(rel_codes[:, :, 2] / ww)) * pixel_x
        dh = F.relu(F.tanh(rel_codes[:, :, 3] / wh)) * pixel_y

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw
        pred_boxes[:, :, 1] = dy + ref_y - dh
        pred_boxes[:, :, 2] = dx + ref_x + dw
        pred_boxes[:, :, 3] = dy + ref_y + dh
        pred_boxes = pred_boxes.clamp_(min=0., max=1.)  #limit the rank

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor.repeat(1, 2)) * self.input_size

    def get_scales(self):
        return (self.boxes[:, :, 2:] - self.boxes[:, :, :2]) * self.input_size

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs, ys = boxes[:, :, 0::2], boxes[:, :, 1::2]
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        ys = torch.nn.functional.interpolate(ys, size=self.patch_pixel, mode='linear', align_corners=True)
        xs, ys = xs.unsqueeze(3).repeat_interleave(self.patch_pixel, dim=3), ys.unsqueeze(2).repeat_interleave(
            self.patch_pixel, dim=2)
        results = torch.stack([xs, ys], dim=-1)
        results = results.reshape(B, -1, 2)#self.patch_count[0] * self.patch_count[1] * self.patch_pixel * self.patch_pixel,
        return results

box = pointwhCoder(input_size=224,patch_count=[1,1],weights=(1.,1.,1.,1.),
                       pts=1,tanh=True,wh_bias=torch.tensor(5./3.).sqrt().log())
if __name__ == "__main__":
    offset=torch.randn((3,192,4))  #1 n m
    off=box(offset,224,[1,1])

