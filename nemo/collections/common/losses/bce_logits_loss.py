# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LogitsType, LogprobsType, LossType, MaskType, NeuralType

__all__ = ["BCEWithLogitsLoss"]


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Serialization, Typing):
    """
    BCELoss
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "logits": NeuralType(["B"] + ["ANY"] * (self._logits_dim - 1), LogitsType()),
            "labels": [NeuralType(["B"] + ["ANY"] * (self._logits_dim - 2), LabelsType())],
            "loss_mask": NeuralType(["B"] + ["ANY"] * (self._logits_dim - 2), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self, logits_ndim=2, weight=None, reduction="mean", ignore_index=-100, pos_weight=None,
    ):
        """
        Args:
            logits_ndim (int): number of dimensions (or rank) of the logits tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
        """
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.FloatTensor(pos_weight)
        # Think about ignore index
        super().__init__(pos_weight=pos_weight, reduction=reduction)
        self._logits_dim = logits_ndim

    @typecheck()
    def forward(self, logits, labels, loss_mask=None):
        """
        Args:
            logits (float): output of the classifier
            labels (float): ground truth labels
        """
        labels = torch.stack(labels)
        labels = labels.t().float()

        return super().forward(logits, labels)
