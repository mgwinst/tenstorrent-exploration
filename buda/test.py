import pybuda
import torch

tt0 = pybuda.TTDevice("tt0")
cpu0 = pybuda.CPUDevice("cpu0")
tt0.place_module(pybuda.PyTorchModule("linear", torch.nn.Linear(64, 64)))

print(tt0, cpu0)

