__author__ = 'sony-w'
__version__ = '1.0'

import torch
import torch.nn as nn

from typing import Union, Sequence


TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]


def embedding_layer(trainable=True, embedding_matrix=None, **kwargs):
    emb_layer = nn.Embedding(**kwargs)
    if embedding_matrix is not None:
        emb_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
    trainable = (embedding_matrix is None) or trainable
    if not trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def get_batch_size(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.size(0)
    else:
        b_s = x[0].size(0)
    return b_s


def get_device(x: TensorOrSequence) -> int:
    if isinstance(x, torch.Tensor):
        b_s = x.device
    else:
        b_s = x[0].device
    return b_s
