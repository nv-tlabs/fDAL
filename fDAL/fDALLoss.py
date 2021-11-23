# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
import torch
from .utils import ConjugateDualFunction

__all__ = ["fDALLoss"]


class fDALLoss(nn.Module):
    def __init__(self, divergence_name, gamma):
        super(fDALLoss, self).__init__()

        self.lhat = None
        self.phistar = None
        self.phistar_gf = None
        self.multiplier = 1.
        self.internal_stats = {}
        self.domain_discriminator_accuracy = -1

        self.gammaw = gamma
        self.phistar_gf = lambda t: ConjugateDualFunction(divergence_name).fstarT(t)
        self.gf = lambda v: ConjugateDualFunction(divergence_name).T(v)

    def forward(self, y_s, y_t, y_s_adv, y_t_adv, K):
        # ---
        #
        #

        v_s = y_s_adv
        v_t = y_t_adv

        if K > 1:
            _, prediction_s = y_s.max(dim=1)
            _, prediction_t = y_t.max(dim=1)

            # This is not used here as a loss, it just a way to pick elements.

            # picking element prediction_s k element from y_s_adv.
            v_s = -F.nll_loss(v_s, prediction_s.detach(), reduction='none')
            # picking element prediction_t k element from y_t_adv.
            v_t = -F.nll_loss(v_t, prediction_t.detach(), reduction='none')

        dst = self.gammaw * torch.mean(self.gf(v_s)) - torch.mean(self.phistar_gf(v_t))

        self.internal_stats['lhatsrc'] = torch.mean(v_s).item()
        self.internal_stats['lhattrg'] = torch.mean(v_t).item()
        self.internal_stats['acc'] = self.domain_discriminator_accuracy
        self.internal_stats['dst'] = dst.item()

        # we need to negate since the obj is being minimized, so min -dst =max dst.
        # the gradient reversar layer will take care of the rest
        return -self.multiplier * dst
