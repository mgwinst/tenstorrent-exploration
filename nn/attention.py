import os
import time
import torch
import ttnn

torch.manual_seed(0)

device_id = 0
dispatch_core_type = ttnn.device.DispatchCoreType.ETH
if "grayskull" in os.environ.get("ARCH_NAME"):
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
device = ttnn.open_device(device_id=device_id, l1_small_size=8192, dispatch_core_type=dispatch_core_type)

ttnn.enable_program_cache(device)

def multi_head_attention(
    hidden_states,
    attention_mask,
    q_w, q_b,
    k_w, k_b,
    v_w, v_b,
    o_w, o_b,
    *,
    num_heads   
):
    
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    q = (hidden_states @ q_w) + q_b
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
    q = ttnn.reshape(q, (batch_size, sequence_size, num_heads, head_size))
    q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)
    q = ttnn.permute(q, (0, 2, 1, 3))

    k = (hidden_states @ k_w) + k_b
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
    k = ttnn.reshape(k, (batch_size, sequence_size, num_heads, head_size))
    k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)
    k = ttnn.permute(k, (0, 2, 3, 1))

    v = (hidden_states @ v_w) + v_b
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
    v = ttnn.reshape(v, (batch_size, sequence_size, num_heads, head_size))
    v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)
    v = ttnn.permute(v, (0, 2, 1, 3))

    attention_scores = (q @ k) * (1 / (head_size**0.5)) + attention_mask
    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ v
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, layout=ttnn.TILE_LAYOUT)
    
    output = (context_layer @ o_w) + o_b

    return output

batch_size = 8
sequence_size = 384
num_heads = 16
head_size = 64
hidden_size = num_heads * head_size

torch_hidden_states = torch.randn((batch_size, sequence_size, hidden_size), dtype=torch.bfloat16)
torch_attention_mask = torch.randn((batch_size, 1, 1, sequence_size), dtype=torch.bfloat16)
torch_query_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
torch_query_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
torch_key_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
torch_key_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
torch_value_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
torch_value_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)
torch_output_weight = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16)
torch_output_bias = torch.randn((hidden_size,), dtype=torch.bfloat16)

hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
query_weight = ttnn.from_torch(torch_query_weight, layout=ttnn.TILE_LAYOUT, device=device)
query_bias = ttnn.from_torch(torch_query_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
key_weight = ttnn.from_torch(torch_key_weight, layout=ttnn.TILE_LAYOUT, device=device)
key_bias = ttnn.from_torch(torch_key_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
value_weight = ttnn.from_torch(torch_value_weight, layout=ttnn.TILE_LAYOUT, device=device)
value_bias = ttnn.from_torch(torch_value_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
output_weight = ttnn.from_torch(torch_output_weight, layout=ttnn.TILE_LAYOUT, device=device)
output_bias = ttnn.from_torch(torch_output_bias, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

if __name__ == "__main__":
    for i in range(10):
        start = time.time()

        multi_head_attention(
            hidden_states,
            attention_mask,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
            output_weight,
            output_bias,
            num_heads=num_heads,
        )

        end = time.time()
        duration = end - start

        print(f"Multi-head attention ran in {duration} seconds for the iteration {i}: ")