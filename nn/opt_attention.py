import os
import time
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

torch.manual_seed(0)

device_id = 0
dispatch_core_type = ttnn.device.DispatchCoreType.ETH
if "grayskull" in os.environ.get("ARCH_NAME"):
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
device = ttnn.open_device(device_id=device_id, l1_small_size=8192, dispatch_core_type=dispatch_core_type)

ttnn.enable_program_cache(device)

def optimized_multi_head_attention(
    hidden_states, attention_mask, 
    fused_qkv_w, fused_qkv_b, 
    output_w, output_b, *,
    num_heads, num_cores_x=12
):
    batch_size, _, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

    fused_qkv_output = ttnn.linear(
        hidden_states,
        fused_qkv_w,
        bias=fused_qkv_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x)
    )

    (q, k, v) = ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads)

    ttnn.deallocate(fused_qkv_output)

    attention_scores = ttnn.matmul(q, k, v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                   coregrid=ttnn.CoreGrid(y=batch_size, x=num_cores_x))
    
    ttnn.deallocate(q)
    ttnn.deallocate(k)

    attention_probs = ttnn.transformer.attention_softmax_(attention_scores, attention_mask=attention_mask, head_size=head_size)

    context_layer = ttnn.matmul(
        attention_probs,
        v,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x)
    )

    ttnn.deallocate(attention_probs)

    context_layer_after_concatenate_heads = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.deallocate(context_layer)

    output = ttnn.linear(
        context_layer_after_concatenate_heads,
        output_w,
        bias=output_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )

    ttnn.deallocate(context_layer_after_concatenate_heads)

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

torch_qkv_weight = torch.cat([torch_query_weight, torch_key_weight, torch_value_weight], dim=-1)
torch_qkv_bias = torch.cat([torch_query_bias, torch_key_bias, torch_value_bias], dim=-1)

qkv_weight = preprocess_linear_weight(torch_qkv_weight.T, dtype=ttnn.bfloat16)
qkv_bias = preprocess_linear_bias(torch_qkv_bias, dtype=ttnn.bfloat16)
output_weight = preprocess_linear_weight(torch_output_weight.T, dtype=ttnn.bfloat16)
output_bias = preprocess_linear_bias(torch_output_bias, dtype=ttnn.bfloat16)

qkv_weight = ttnn.to_device(qkv_weight, device)
qkv_bias = ttnn.to_device(qkv_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output_weight = ttnn.to_device(output_weight, device)
output_bias = ttnn.to_device(output_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)


for i in range(10):
    
    start = time.time()
    hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
    optimized_output = optimized_multi_head_attention(
        hidden_states,
        attention_mask,
        qkv_weight,
        qkv_bias,
        output_weight,
        output_bias,
        num_heads=num_heads,
    )
    end = time.time()
    duration = end - start 

    print(f"Optimized multi-head attention ran in {duration} seconds for the iteration {i}")


ttnn.close_device(device)

