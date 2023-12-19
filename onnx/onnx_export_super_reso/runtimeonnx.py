# ref: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import onnx
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.init as init

onnx_model = onnx.load("../../data/super_resolution.onnx")
print(onnx.checker.check_model(onnx_model))
batch_size = 1
ort_session = onnxruntime.InferenceSession("../../data/super_resolution.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

torch_model = SuperResolutionNet(upscale_factor=3)
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# todo: here
# todo: here
# todo: here

# import onnxruntime
# # Some standard imports
# import numpy as np

# from torch import nn
# import torch.utils.model_zoo as model_zoo
# import torch.onnx
# batch_size = 1

# net_path = "../../data/super_resolution.onnx"
# ort_session = onnxruntime.InferenceSession(net_path, providers=["CPUExecutionProvider"])

# class SuperResolutionNet(nn.Module):
#     def __init__(self, upscale_factor, inplace=False):
#         super(SuperResolutionNet, self).__init__()

#         self.relu = nn.ReLU(inplace=inplace)
#         self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
#         self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
#         self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
#         self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

#         self._initialize_weights()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.pixel_shuffle(self.conv4(x))
#         return x

#     def _initialize_weights(self):
#         init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
#         init.orthogonal_(self.conv4.weight)

# # Create the super-resolution model by using the above model definition.
# torch_model = SuperResolutionNet(upscale_factor=3)

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
# torch_out = torch_model(x)

# # compute ONNX Runtime output prediction
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")
