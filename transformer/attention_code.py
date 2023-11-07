import torch
import torch.nn.functional as F
# ref: link https://peterbloem.nl/blog/transformers
# assume we have some tensor x with size (b, t, k)
x = ...

# todo: investigate this
# todo: investigate this
# todo: investigate this


raw_weights = torch.bmm(x, x.transpose(1, 2))