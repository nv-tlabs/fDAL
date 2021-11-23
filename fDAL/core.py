# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch.nn as nn
from .fDALLoss import fDALLoss
from .utils import WarmGRL
import torch
import copy


class fDALLearner(nn.Module):
    def __init__(self, backbone, taskhead, taskloss, divergence, bootleneck=None, reg_coef=1, n_classes=-1,
                 aux_head=None,
                 grl_params=None):
        """
        fDAL Learner.
        :param backbone: z=backbone(input). Thus backbone must be nn.Module. (i.e Usually resnet without last f.c layers).
        :param taskhead: prediction = taskhead(z). Thus taskhead must be nn.Module *(e.g The last  f.c layers of Resnet)
        :param taskloss: he loss used to trained the model. i.e nn.CrossEntropy()
        :param divergence: divergence name (i.e pearson, jensen).
        :param bootleneck: (optional) a bootleneck layer after feature extractor and before the classifier.
        :param reg_coef: the coefficient to weight the domain adaptation loss (fDAL gamma coefficient).
        :param n_classes: if output is categorical then the number of classes. if <=1 will create a global discriminator.
        :param aux_head: (optional) if specified with use the provided head as the domain-discriminator. If not will create it based on tashhead as described in the paper.
        :param grl_params: dict with grl_params.
        """

        super(fDALLearner, self).__init__()
        self.backbone = backbone
        self.taskhead = taskhead
        self.taskloss = taskloss
        self.bootleneck = bootleneck
        self.n_classes = n_classes
        self.reg_coeff = reg_coef
        self.auxhead = aux_head if aux_head is not None else self.build_aux_head_()

        self.fdal_divhead = fDALDivergenceHead(divergence, self.auxhead, n_classes=self.n_classes,
                                               grl_params=grl_params,
                                               reg_coef=reg_coef)

    def build_aux_head_(self):
        # fDAL recommends the same architecture for both h, h'
        auxhead = copy.deepcopy(self.taskhead)
        if self.n_classes == -1:
            # creates a global discriminator, fall back to DANN in most cases. useful for multihead networks.
            aux_linear = auxhead[-1]
            auxhead[-1] = nn.Sequential(
                nn.Linear(aux_linear.in_features, 1)
            )

        # different initialization.
        auxhead.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)
        return auxhead

    def forward(self, x, y, src_size=-1, trg_size=-1):
        """
        :param x: tensor or tuple containing source and target input tensors.
        :param y: tensor or tuple containing source and target label tensors. (if unsupervised adaptation is a tensor with labels for source)
        :param src_size: src_size if specified. otherwise computed from input tensors
        :param trg_size: trg_size if specified. otherwise computed from input tensors

        :return: returns a tuple(tensor,dict). e.g. total_loss, {"pred_s": outputs_src, "pred_t": outputs_tgt, "taskloss": task_loss}

        """
        if isinstance(x, tuple):
            # assume x=x_source, x_target
            src_size = x[0].shape[0]
            trg_size = x[1].shape[0]
            x = torch.cat((x[0], x[1]), dim=0)

        y_s = y
        y_t = None

        if isinstance(y, tuple):
            # assume y=y_source, y_target, otherwise assume y=y_source
            # warnings.warn_explicit('using target data')
            y_s = y[0]
            y_t = y[1]

        f = self.backbone(x)
        f = self.bootleneck(f) if self.bootleneck is not None else f

        net_output = self.taskhead(f)

        # splitting source and target features
        f_source = f.narrow(0, 0, src_size)
        f_tgt = f.narrow(0, src_size, trg_size)

        # h(g(x))
        outputs_src = net_output.narrow(0, 0, src_size)
        outputs_tgt = net_output.narrow(0, src_size, trg_size)

        # computing losses....

        # task loss in source...
        task_loss = self.taskloss(outputs_src, y_s)

        # task loss in target if labels provided. Warning!. Only on semi-sup adaptation.
        task_loss += 0.0 if y_t is None else self.taskloss(outputs_tgt, y_t)

        fdal_loss = 0.0
        if self.reg_coeff > 0.:
            # adaptation
            fdal_loss = self.fdal_divhead(f_source, f_tgt, outputs_src, outputs_tgt)

            # together
            total_loss = task_loss + fdal_loss
        else:
            total_loss = task_loss

        return total_loss, {"pred_s": outputs_src, "pred_t": outputs_tgt, "taskloss": task_loss, "fdal_loss": fdal_loss,
                            "fdal_src": self.fdal_divhead.internal_stats["lhatsrc"],
                            "fdal_trg": self.fdal_divhead.internal_stats["lhattrg"]}

    def get_reusable_model(self, pack=False):
        """
        Returns the usable parts of the model. For example backbone and taskhead. ignore the rest.

        :param pack: if set to True. will return a model that looks like taskhead( backbone(input)). Useful for inference.
        :return: nn.Module  or tuple of nn.Modules
        """
        if pack is True:
            return nn.Sequential(self.backbone, self.taskhead)
        return self.backbone, self.taskhead


class fDALDivergenceHead(nn.Module):
    def __init__(self, divergence_name, aux_head, n_classes, grl_params=None, reg_coef=1.):
        """
        :param divergence_name: divergence name (i.e pearson, jensen).
        :param aux_head: the auxiliary head refer to paper fig 1.
        :param n_classes:  if output is categorical then the number of classes. if <=1 will create a global discriminator.
        :param grl_params:  dict with grl_params.
        :param reg_coef: regularization coefficient. default 1.
        """
        super(fDALDivergenceHead, self).__init__()
        self.grl = WarmGRL(auto_step=True) if grl_params is None else WarmGRL(**grl_params)
        self.aux_head = aux_head
        self.fdal_loss = fDALLoss(divergence_name, gamma=1.0)
        self.internal_stats = self.fdal_loss.internal_stats
        self.n_classes = n_classes
        self.reg_coef = reg_coef

    def forward(self, features_s, features_t, pred_src, pred_trg) -> torch.Tensor:
        """
        :param features_s: features extracted by backbone on source data.
        :param features_t: features extracted by backbone on target data.
        :param pred_src: prediction on src data (for classification tasks should be N,n_classes (logits))
        :param pred_trg: prediction on trg data (for classification tasks should be N,n_classes (logits))
        :return: fdal loss
        """

        f = self.grl(torch.cat((features_s, features_t), dim=0))
        src_size = features_s.shape[0]
        trg_size = features_t.shape[0]

        aux_output_f = self.aux_head(f)

        # h'(g(x)) auxiliary head output on source and target respectively.
        y_s_adv = aux_output_f.narrow(0, 0, src_size)
        y_t_adv = aux_output_f.narrow(0, src_size, trg_size)

        loss = self.fdal_loss(pred_src, pred_trg, y_s_adv, y_t_adv, self.n_classes)
        self.internal_stats = self.fdal_loss.internal_stats  # for debugging.

        return self.reg_coef * loss
