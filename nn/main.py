import torch
import ttnn

torch.manual_seed(0)

device_id = 0
device = ttnn.open_device(device_id=device_id)



ttnn.close_device(device)