# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
from mmdet.models import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.backbones import ResNet
import torch.nn.functional as F
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class OSSAPrototypeHook(Hook):
    def before_run(self, runner) -> None:
        print("Prototypes Loaded")
        runner.model.backbone.load_prototypes()

    def after_run(self, runner) -> None:
        runner.model.backbone.calculate_and_set_prototypes()


@MODELS.register_module()
class ResNetOSSA(ResNet):
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=False),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None,
                 aug_stages=[0, 1],
                 aug_prob=[0.5],
                 aug_intensity=0.75,
                 make_prototypes=False,
                 num_shots=1):
        super(ResNetOSSA, self).__init__(
            depth=depth,
            in_channels=in_channels,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            plugins=plugins,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            pretrained=pretrained,
            init_cfg=init_cfg
        )
        self.aug_stages = aug_stages
        self.aug_prob = aug_prob
        self.aug_intensity = aug_intensity
        self.make_prototypes = make_prototypes
        self.num_shots = num_shots

        self.memory_banks = [[] for _ in self.aug_stages]
        self.prototype_means = []
        self.prototype_stds = []

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        p = torch.rand(1).item()
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if self.training and i in self.aug_stages and self.make_prototypes:
                self.update_memory_banks(x, i)
            if self.training and i in self.aug_stages and p < self.aug_prob and not self.make_prototypes:
                x = self.ossa(x, i)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


    def ossa(self, x, stage_index):
        B, C, H, W = x.shape
        device = x.device

        # Compute the mean and standard deviation for each feature map
        mu = x.mean(dim=[2, 3], keepdim=True)  # Shape: BxCx1x1
        std = x.std(dim=[2, 3], keepdim=True)

        # Sample alpha and beta from a Gaussian distribution
        alpha = torch.normal(mean=1.0, std=self.aug_intensity, size=(B, C, 1, 1), device=device)
        beta = torch.normal(mean=1.0, std=self.aug_intensity, size=(B, C, 1, 1), device=device)

        target_mu = self.prototype_means[stage_index]
        target_std = self.prototype_stds[stage_index]

        new_mu = target_mu * alpha
        new_std = target_std * beta

        normalized_x = (x - mu) / (std + 1e-7)

        perturbed_x = new_std * normalized_x + new_mu

        return perturbed_x

    def update_memory_banks(self, feature_map, stage_index):
        # Calculate the mean and standard deviation for the batch
        batch_mean = feature_map.mean([2, 3], keepdim=True)
        batch_std = feature_map.std([2, 3], keepdim=True)

        # Convert the batch of means and stds into lists of individual tensors
        individual_means = [batch_mean[i] for i in range(batch_mean.size(0))]
        individual_stds = [batch_std[i] for i in range(batch_std.size(0))]

        # Extend the memory bank with tuples of (mean, std) for each feature map
        self.memory_banks[stage_index].extend(zip(individual_means, individual_stds))

        # Clamp the memory bank
        self.memory_banks[stage_index] = self.memory_banks[stage_index][:self.num_shots]


    def calculate_and_set_prototypes(self):
        if not self.make_prototypes:
            return

        self.prototype_means = []  # Reset or ensure it's a list
        self.prototype_stds = []  # Reset or ensure it's a list

        for i in range(len(self.aug_stages)):
            if not self.memory_banks[i]:  # If there's nothing to process, skip this iteration
                continue

            means, stds = zip(*self.memory_banks[i])
            all_means = torch.stack(means, dim=0)
            all_stds = torch.stack(stds, dim=0)
            prototype_mean = torch.mean(all_means, dim=0)
            prototype_std = torch.mean(all_stds, dim=0)

            # Append the new prototype to the lists
            self.prototype_means.append(prototype_mean)
            self.prototype_stds.append(prototype_std)

        # After processing, consider saving the prototypes
        self.save_prototype_to_file('prototype_means.pt', self.prototype_means)
        self.save_prototype_to_file('prototype_stds.pt', self.prototype_stds)

        print("Prototypes saved")

    def save_prototype_to_file(self, filename, prototypes):
        # Check if prototype is not None
        if prototypes is not None:
            torch.save(prototypes, filename)
        else:
            print("No prototype to save.")

    def load_prototypes(self):
        if self.make_prototypes or not self.training:
            return
        # Check if the file exists
        filename = 'prototype_means.pt'
        if os.path.isfile(filename):
            # Load the prototype from the file
            self.prototype_means = torch.load(filename)
            print("Prototype mean loaded from file.")
        filename = 'prototype_stds.pt'
        if os.path.isfile(filename):
            # Load the prototype from the file
            self.prototype_stds = torch.load(filename)
            print("Prototype std loaded from file.")



