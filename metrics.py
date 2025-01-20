#!/usr/bin/env python3
# @file      metrics.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class ClassificationMetrics(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes)).long()

    # def compute_confusion_matrix(self, pred_labels: torch.Tensor, gt_labels: torch.Tensor):
    def compute_confusion_matrix(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor):
        pred_softmax = F.softmax(pred_logits, dim=1)
        pred_labels = torch.argmax(pred_softmax, axis=1).long()
        gt_labels = gt_labels.long()

        idxs = torch.stack([pred_labels, gt_labels], dim=0)
        ones = torch.ones((idxs.shape[-1])).type_as(gt_labels)
        self.conf_matrix = self.conf_matrix.index_put_(tuple(idxs), ones, accumulate=True)

    def getStats(self, confusion_matrix):
        # we only care about moving class
        tp = confusion_matrix.diag()[1]
        fp = confusion_matrix.sum(dim=1)[1] - tp
        fn = confusion_matrix.sum(dim=0)[1] - tp
        return tp, fp, fn

    def getIoU(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def getacc(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        total_tp = tp.sum()
        total = tp.sum() + fp.sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean

    def getStaticIoU(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        tn = confusion_matrix.diag()[0]
        intersection = tn
        union = tn + fp + fn + 1e-15
        iou = intersection / union
        return iou

    def getStaticAcc(self, confusion_matrix):
        tp, fp, fn = self.getStats(confusion_matrix)
        tn = confusion_matrix.diag()[0]
        total_tn = tn.sum()
        total = tn.sum() + fn.sum() + 1e-15
        acc_mean = total_tn / total
        return acc_mean

