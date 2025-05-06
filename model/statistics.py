# 统计模型参数
import torch
import torch.nn as nn
from torchinfo import summary
from model.mda_rsm import MDA_RSM
from config.hyperparameter_mdscans_WHU import ph

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移动到目标设备
model = MDA_RSM(
    dims=ph.dims,
    depths=ph.depths,
    ssm_d_state=ph.ssm_d_state,
    ssm_dt_rank=ph.ssm_dt_rank,
    ssm_ratio=ph.ssm_ratio,
    mlp_ratio=ph.mlp_ratio,
    scan_types=ph.scan_types,
    per_scan_num=ph.per_scan_num,
).to(device)

# 创建输入张量
input_tensor = torch.randn(8, 3, 512, 512).to(device)

# 1. Complexity (M) 和 GFLOPs
try:
    # 通过 torchinfo 获取 FLOPs 和其他统计
    model_info = summary(
        model, input_size=(8, 3, 512, 512), verbose=0, device=device.type
    )
    complexity_m = model_info.total_mult_adds / 1e6  # 单位：百万 (M)
    gflops = model_info.total_mult_adds / 1e9  # 单位：GFLOPs
    print(f"Complexity (M): {complexity_m:.4f}")
    print(f"GFLOPs: {gflops:.4f}")
except Exception as e:
    print(f"Error calculating FLOPs: {e}")

# 2. Memory (MB)
try:
    # 计算输入张量的内存
    input_memory = input_tensor.element_size() * input_tensor.nelement()  # 输入占用内存
    model.eval()  # 模型设为评估模式，确保一致性
    with torch.no_grad():
        output = model(input_tensor)  # 前向传播，得到输出张量
    output_memory = output.element_size() * output.nelement()  # 输出占用内存

    # 计算模型参数的内存
    param_memory = sum(p.nelement() * p.element_size() for p in model.parameters())

    # 总内存（单位：MB）
    total_memory = (input_memory + output_memory + param_memory) / (1024**2)
    print(f"Memory required (MB): {total_memory:.4f}")
except Exception as e:
    print(f"Error calculating memory: {e}")

# 3. Param (M)
try:
    # 模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    param_m = total_params / 1e6  # 转换为百万级
    print(f"Param (M): {param_m:.4f}")
except Exception as e:
    print(f"Error calculating parameters: {e}")
