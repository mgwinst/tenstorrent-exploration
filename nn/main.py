import torch
import ttnn

torch.manual_seed(0)

device_id = 0
device = ttnn.open_device(device_id=device_id)

device.enable_program_cache()

m = 2048
n = 2048
k = 2048

a = ttnn.from_torch(torch.randn(m, k), dtype=ttnn.bfloat16)
b = ttnn.from_torch(torch.randn(k, n), dtype=ttnn.bfloat16)

a = ttnn.to_device(a, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
b = ttnn.to_device(b, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)
b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)

c = a @ b

print(c)

ttnn.close_device(device)
