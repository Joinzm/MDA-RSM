# Import necessary libraries
import torch
import torch.nn as nn
from torchinfo import summary
from model.mda_rsm import MDA_RSM
from config.hyperparameter_mdscans_WHU import ph

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and move it to the target device
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

# Create the input tensor
input_tensor = torch.randn(8, 3, 512, 512).to(device)

# 1. Complexity (M) and GFLOPs
try:
    # Use torchinfo to get FLOPs and other statistics
    model_info = summary(
        model, input_size=(8, 3, 512, 512), verbose=0, device=device.type
    )
    complexity_m = model_info.total_mult_adds / 1e6  # Unit: Million (M)
    gflops = model_info.total_mult_adds / 1e9  # Unit: GFLOPs
    print(f"Complexity (M): {complexity_m:.4f}")
    print(f"GFLOPs: {gflops:.4f}")
except Exception as e:
    print(f"Error calculating FLOPs: {e}")

# 2. Memory (MB)
try:
    # Calculate memory for the input tensor
    input_memory = input_tensor.element_size() * input_tensor.nelement()  # Input memory
    model.eval()  # Set the model to evaluation mode for consistency
    with torch.no_grad():
        output = model(input_tensor)  # Perform forward pass to get the output tensor
    output_memory = output.element_size() * output.nelement()  # Output memory

    # Calculate memory for model parameters
    param_memory = sum(p.nelement() * p.element_size() for p in model.parameters())

    # Total memory (Unit: MB)
    total_memory = (input_memory + output_memory + param_memory) / (1024**2)
    print(f"Memory required (MB): {total_memory:.4f}")
except Exception as e:
    print(f"Error calculating memory: {e}")

# 3. Param (M)
try:
    # Calculate the number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    param_m = total_params / 1e6  # Convert to millions
    print(f"Param (M): {param_m:.4f}")
except Exception as e:
    print(f"Error calculating parameters: {e}")
