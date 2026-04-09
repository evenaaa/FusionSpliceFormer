#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/28 16:26
# @Author  : even
# @File    : dataset.py
from torch.utils.data import Dataset

class FSFDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        if not isinstance(self.inputs, dict):
            raise TypeError("inputs must be dict")
        if len(self.inputs) == 0:
            raise ValueError("inputs null!")
        self.dataset_len = len(next(iter(self.inputs.values())))
        self._check_input_length_consistency()

    def _check_input_length_consistency(self):
        base_len = self.dataset_len
        for key, tensor in self.inputs.items():
            if len(tensor) != base_len:
                raise ValueError(f"length of tensor and key not equal")

    def __getitem__(self, index):
        single_sample = {}
        for key in self.inputs:
            single_sample[key] = self.inputs[key][index]
        return single_sample

    def __len__(self):
        return self.dataset_len
