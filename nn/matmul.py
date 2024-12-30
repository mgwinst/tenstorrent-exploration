import os
import torch
import ttnn

torch.manual_seed(0)

device_id = 0
device = ttnn.open_device(device_id=device_id)
ttnn.enable_program_cache(device)

m = 1024
k = 1024
n = 1024

torch_a = torch.randn((m, k), dtype=torch.bfloat16)
torch_b = torch.randn((k, n), dtype=torch.bfloat16)

a = ttnn.from_torch(torch_a)
b = ttnn.from_torch(torch_b)

a = ttnn.to_device(a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
b = ttnn.to_device(b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

# necessary to restate this?
output = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=8, x=8))

ttnn.close_device(device)
