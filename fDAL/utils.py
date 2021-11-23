# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Implementation of WarmGRL obtained from https://github.com/thuml/Transfer-Learning-Library/blob/8a718bb16cca540a63907e645726b9da08f0bee1/dalib/modules/grl.py
# MIT License.

# The MIT License (MIT)
# Copyright (c) 2020 JunguangJiang
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from typing import Optional, Any, Tuple
import numpy as np
import torch.nn as nn
from torch.autograd import Function
import torch
from math import log
from torch.functional import F

__all__ = ['WarmGRL', 'ConjugateDualFunction']


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmGRL(nn.Module):
    """Gradient Reverse Layer with warm start
        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = True):
        super(WarmGRL, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.coeff_log = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        self.coeff_log = coeff
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

    def log_status(self):
        params = {f'{k}': v for k, v in self.__dict__.items() if isinstance(v, (str, int, float))}
        return params

#Inspired by the work of https://arxiv.org/abs/1606.00709.
class ConjugateDualFunction:

    def __init__(self, divergence_name, gamma=4):
        self.f_div_name = divergence_name
        self.gamma = gamma

    def T(self, v):
        """T(v)"""

        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return v
        elif self.f_div_name == "klrev":
            return -torch.exp(v)
        elif self.f_div_name == "pearson":
            return v
        elif self.f_div_name == "neyman":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "hellinger":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "jensen":
            return log(2.0) - F.softplus(-v)
        elif self.f_div_name == "gammajensen":
            return -self.gamma * log(self.gamma) - F.softplus(-v)
        else:
            raise ValueError("Unknown divergence.")

    def fstarT(self, v):
        """f^*(T(v))"""

        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "klrev":
            return -1.0 - v
        elif self.f_div_name == "pearson":
            return 0.25 * v * v + v
        elif self.f_div_name == "neyman":
            return 2.0 - 2.0 * torch.exp(0.5 * v)
        elif self.f_div_name == "hellinger":
            return torch.exp(-v) - 1.0
        elif self.f_div_name == "jensen":
            return F.softplus(v) - log(2.0)
        elif self.f_div_name == "gammajensen":
            gf = lambda v_: -self.gamma * log(self.gamma) - F.softplus(-v_)
            return -torch.log(self.gamma + 1. - self.gamma * torch.exp(gf(v))) / self.gamma
        else:
            raise ValueError("Unknown divergence.")
