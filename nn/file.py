import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch_tensor_a = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
torch_tensor_b = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)

input_tensor_a = ttnn.from_torch(torch_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

output_tensor = input_tensor_a @ input_tensor_b

print(f"shape: {output_tensor.shape}")
print(f"dtype: {output_tensor.dtype}")
print(f"layout: {output_tensor.layout}")

print(output_tensor)

ttnn.close_device(device)

