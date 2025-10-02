#!/usr/bin/env python3
"""
Utility functions for Voxtral audio_encoder testing.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
import json


def create_synthetic_audio_input(
    batch_size: int = 1, 
    n_mels: int = 80, 
    seq_length: int = 3000,
    device: str = "cpu"
) -> torch.Tensor:
    """Create synthetic audio features for testing."""
    return torch.randn(batch_size, n_mels, seq_length, device=device)


def save_test_metadata(test_dir: str, metadata: Dict[str, Any]):
    """Save test metadata to JSON file."""
    metadata_file = os.path.join(test_dir, "test_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_test_metadata(test_dir: str) -> Optional[Dict[str, Any]]:
    """Load test metadata from JSON file."""
    metadata_file = os.path.join(test_dir, "test_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def validate_tensor_output(tensor: torch.Tensor) -> bool:
    """Validate tensor output for common issues."""
    if tensor is None:
        return False
    
    # Check for NaN or Inf values
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    
    # Check if tensor has reasonable shape
    if tensor.numel() == 0:
        return False
        
    return True


def print_tensor_stats(tensor: torch.Tensor, name: str = "tensor"):
    """Print detailed statistics about a tensor."""
    if tensor is None:
        print(f"{name}: None")
        return
        
    tensor_np = tensor.detach().cpu().numpy()
    
    print(f"{name} Statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Min: {tensor_np.min():.6f}")
    print(f"  Max: {tensor_np.max():.6f}")
    print(f"  Mean: {tensor_np.mean():.6f}")
    print(f"  Std: {tensor_np.std():.6f}")
    print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
    print(f"  Inf count: {torch.isinf(tensor).sum().item()}")


def compare_tensor_shapes(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """Compare shapes of two tensors."""
    if tensor1 is None or tensor2 is None:
        return False
    return tensor1.shape == tensor2.shape


def get_system_info() -> Dict[str, Any]:
    """Get system information for testing metadata."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
    return info


if __name__ == "__main__":
    # Quick test of utility functions
    print("Testing utility functions...")
    
    # Test synthetic audio input creation
    audio_input = create_synthetic_audio_input()
    print_tensor_stats(audio_input, "synthetic_audio")
    
    # Test system info
    sys_info = get_system_info()
    print(f"System info: {sys_info}")
    
    print("Utility functions test completed!")