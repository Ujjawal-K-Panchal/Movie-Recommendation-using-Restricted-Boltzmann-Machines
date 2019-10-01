# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:23:00 2019

@author: Uchiha Madara
"""

import torch
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        z = wx + self.a
        p_h_given_v = torch.sigmoid(z)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, x):
        wx = torch.mm(x, self.W)
        z = wx + self.b
        p_v_given_h = torch.sigmoid(z)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk),0)
