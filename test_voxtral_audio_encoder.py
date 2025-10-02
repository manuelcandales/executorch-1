#!/usr/bin/env python3
"""
Test script for Voxtral audio_encoder component with ExecuTorch Metal backend.
Compares outputs between PyTorch reference and ExecuTorch Metal implementation.

Usage:
    python test_voxtral_audio_encoder.py --mode export     # Export the model
    python test_voxtral_audio_encoder.py --mode test      # Run inference test
    python test_voxtral_audio_encoder.py --mode compare   # Export + Test + Compare
"""

import argparse
import logging
import os
import sys
from typing import Tuple, Dict, Any
import tempfile
import shutil

import torch
import numpy as np
from transformers import AutoConfig, AutoProcessor

from optimum.exporters.executorch.tasks.multimodal_text_to_text import load_multimodal_text_to_text_model
from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
from executorch.exir import to_edge_transform_and_lower

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly send to stdout
    ]
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
DEVICE = "mps"  # Use Metal Performance Shaders on macOS


class VoxtralAudioEncoderTester:
    """Test harness for Voxtral audio_encoder component."""

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE):
        self.model_id = model_id
        self.device = device
        self.config = None
        self.processor = None
        self.model = None
        self.audio_encoder_ep = None
        self.test_data_dir = "voxtral_test_data"

        # Create test data directory
        os.makedirs(self.test_data_dir, exist_ok=True)

    def setup_model_and_data(self):
        """Initialize model components and test data."""
        logger.info(f"Loading Voxtral model: {self.model_id}")

        # Load config and processor
        self.config = AutoConfig.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Load the full model using the optimum library
        logger.info("Loading multimodal model...")
        self.model = load_multimodal_text_to_text_model(
            self.model_id,
            device=self.device,
            use_custom_sdpa=False,
            use_custom_kv_cache=False,
        )

        logger.info("Model loaded successfully")

    def generate_test_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate test inputs for the audio encoder."""
        logger.info("Generating test inputs...")

        # Create a conversation with audio input
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        # Process the conversation
        inputs = self.processor.apply_chat_template(conversation)

        # Move inputs to device
        for name in inputs:
            inputs[name] = inputs[name].to(self.device)

        logger.info(f"Generated test inputs with keys: {list(inputs.keys())}")
        if "input_features" in inputs:
            logger.info(f"Audio input_features shape: {inputs['input_features'].shape}")

        return inputs

    def export_audio_encoder(self):
        """Export the audio_encoder component to ExecuTorch Metal."""
        logger.info("Exporting audio_encoder component...")

        if self.model is None:
            self.setup_model_and_data()

        # Export the model components
        ep = self.model.export()
        self.audio_encoder_ep = ep["audio_encoder"]

        logger.info("Starting ExecuTorch Metal export...")

        # Define output paths
        output_pte_path = os.path.join(self.test_data_dir, "voxtral_audio_encoder_metal.pte")
        output_data_dir = os.path.join(self.test_data_dir, "voxtral_audio_encoder_metal_data")

        # Create output directory
        os.makedirs(output_data_dir, exist_ok=True)

        try:
            # Convert to Edge dialect with Metal partitioner
            logger.info("Converting to Edge dialect with Metal partitioner...")
            edge_program = to_edge_transform_and_lower(
                self.audio_encoder_ep, partitioner=[MetalPartitioner([])]
            )
            logger.info("Edge dialect conversion completed")

            # Convert to ExecuTorch program
            logger.info("Converting to ExecuTorch program...")
            executorch_program = edge_program.to_executorch()
            logger.info("ExecuTorch program conversion completed")

            # Save the compiled .pte program
            logger.info(f"Saving .pte file to: {output_pte_path}")
            with open(output_pte_path, "wb") as file:
                file.write(executorch_program.buffer)

            # Save tensor data
            logger.info(f"Saving tensor data to: {output_data_dir}")
            logger.info(f"Number of tensor data files: {len(executorch_program._tensor_data)}")
            executorch_program.write_tensor_data_to_file(output_data_dir)

            logger.info("ExecuTorch Metal export completed successfully!")
            return output_pte_path, output_data_dir

        except Exception as e:
            logger.error(f"Error during ExecuTorch export: {e}")
            raise

    def run_pytorch_reference(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run PyTorch reference implementation."""
        logger.info("Running PyTorch reference implementation...")

        if self.model is None:
            self.setup_model_and_data()

        if "input_features" not in inputs:
            logger.error("No audio input_features found in inputs")
            raise ValueError("Audio input_features required for audio encoder test")

        input_features = inputs["input_features"]
        logger.info(f"Original input_features shape: {input_features.shape}")

        # Create "all ones" inputs to match what executor_runner uses
        # This ensures fair comparison between PyTorch and ExecuTorch Metal backend
        ones_input_features = torch.ones_like(input_features, device=input_features.device, dtype=input_features.dtype)
        logger.info(f"Using synthetic 'all ones' inputs for fair comparison with executor_runner")
        logger.info(f"Synthetic input shape: {ones_input_features.shape}, dtype: {ones_input_features.dtype}")

        # Export and run the audio encoder
        if self.audio_encoder_ep is None:
            ep = self.model.export()
            self.audio_encoder_ep = ep["audio_encoder"]

        # Run the exported PyTorch model with synthetic inputs
        with torch.no_grad():
            pytorch_output = self.audio_encoder_ep.module()(input_features=ones_input_features)

        logger.info(f"PyTorch output shape: {pytorch_output.shape}")
        logger.info(f"PyTorch output dtype: {pytorch_output.dtype}")

        # Save PyTorch reference output
        output_file = os.path.join(self.test_data_dir, "pytorch_reference_output.txt")
        self._save_tensor_to_file(pytorch_output, output_file)

        return pytorch_output

    def run_executorch_inference(self) -> torch.Tensor:
        """Run inference using ExecuTorch Metal backend."""
        logger.info("Running ExecuTorch Metal inference with executor_runner...")

        # Paths to the exported ExecuTorch files
        pte_path = os.path.join(self.test_data_dir, "voxtral_audio_encoder_metal.pte")
        data_dir = os.path.join(self.test_data_dir, "voxtral_audio_encoder_metal_data")

        if not os.path.exists(pte_path):
            raise FileNotFoundError(f"ExecuTorch program file not found: {pte_path}")

        executorch_output = self._run_executorch_subprocess(pte_path, data_dir)

        logger.info(f"ExecuTorch output shape: {executorch_output.shape}")
        logger.info(f"ExecuTorch output dtype: {executorch_output.dtype}")

        # Save ExecuTorch output
        output_file = os.path.join(self.test_data_dir, "executorch_metal_output.txt")
        self._save_tensor_to_file(executorch_output, output_file)

        return executorch_output

    def _run_executorch_subprocess(self, pte_path: str, data_dir: str) -> torch.Tensor:
        """Run ExecuTorch inference using subprocess with executor_runner."""
        import subprocess

        logger.info("Using executor_runner with ExecuTorch Metal backend")
        logger.info("NOTE: executor_runner uses synthetic inputs (all ones) for testing")

        executor_runner = "./cmake-out/executor_runner"

        # Create output directory for executor_runner
        os.makedirs("aoti_debug_data", exist_ok=True)

        # Build command
        cmd = [executor_runner, "--model_path", pte_path, "--data_path", data_dir]

        logger.info(f"Running: {' '.join(cmd)}")

        # Run executor_runner
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=os.getcwd())

        if result.returncode != 0:
            logger.error(f"executor_runner failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            logger.error(f"stdout: {result.stdout}")
            raise RuntimeError(f"executor_runner failed: {result.stderr}")

        logger.info("executor_runner completed successfully")
        logger.info(f"stdout: {result.stdout}")

        # Read and parse output
        output_file = "aoti_debug_data/final_runtime_output.txt"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")

        with open(output_file, 'r') as f:
            output_content = f.read().strip()

        if not output_content:
            raise RuntimeError("Output file is empty")

        # Parse comma-separated values
        output_values = [float(x.strip()) for x in output_content.split(',') if x.strip()]
        output_tensor = torch.tensor(output_values, dtype=torch.float32)

        # Try to reshape to expected audio encoder output shape [batch, sequence, hidden_dim]
        if len(output_values) > 0:
            batch_size = 1
            hidden_dim = 384  # Common for Voxtral Mini
            seq_length = len(output_values) // hidden_dim

            if len(output_values) % hidden_dim == 0 and seq_length > 0:
                output_tensor = output_tensor.reshape(batch_size, seq_length, hidden_dim)
                logger.info(f"Reshaped output to: {output_tensor.shape}")

        logger.info(f"Parsed executor_runner output: {output_tensor.shape}")
        return output_tensor

    def _save_tensor_to_file(self, tensor: torch.Tensor, filepath: str):
        """Save tensor values to a text file in CSV format."""
        tensor_np = tensor.detach().cpu().numpy().flatten()
        with open(filepath, 'w') as f:
            f.write(','.join(map(str, tensor_np.tolist())))
        logger.info(f"Saved tensor data to: {filepath}")

    def compare_outputs(self, pytorch_output: torch.Tensor, executorch_output: torch.Tensor) -> Tuple[float, float, bool]:
        """Compare PyTorch and ExecuTorch outputs."""
        logger.info("Comparing PyTorch and ExecuTorch outputs...")

        # Convert to numpy for comparison
        pytorch_np = pytorch_output.detach().cpu().numpy()
        executorch_np = executorch_output.detach().cpu().numpy()

        # Check shapes match
        if pytorch_np.shape != executorch_np.shape:
            logger.error(f"Output shapes don't match: PyTorch {pytorch_np.shape} vs ExecuTorch {executorch_np.shape}")
            return None, None, False

        # Calculate differences
        abs_diff = np.abs(pytorch_np - executorch_np)
        max_atol = np.max(abs_diff)
        mean_atol = np.mean(abs_diff)

        # Calculate relative differences
        eps = 1e-8
        denominator = np.maximum(
            np.maximum(np.abs(pytorch_np), np.abs(executorch_np)), eps
        )
        rel_diff = abs_diff / denominator
        max_rtol = np.max(rel_diff)
        mean_rtol = np.mean(rel_diff)

        # Check if outputs are close within common tolerances
        is_close_1e5 = np.allclose(pytorch_np, executorch_np, atol=1e-5, rtol=1e-5)
        is_close_1e6 = np.allclose(pytorch_np, executorch_np, atol=1e-6, rtol=1e-6)
        is_close_1e4 = np.allclose(pytorch_np, executorch_np, atol=1e-4, rtol=1e-4)

        # Print comparison results
        logger.info("=" * 60)
        logger.info("COMPARISON RESULTS:")
        logger.info(f"Output shapes: {pytorch_np.shape}")
        logger.info(f"Total elements: {pytorch_np.size}")
        logger.info("-" * 40)
        logger.info(f"Max Absolute Tolerance (atol): {max_atol:.10f}")
        logger.info(f"Mean Absolute Tolerance: {mean_atol:.10f}")
        logger.info(f"Max Relative Tolerance (rtol): {max_rtol:.10f}")
        logger.info(f"Mean Relative Tolerance: {mean_rtol:.10f}")
        logger.info("-" * 40)
        logger.info(f"Close within atol=1e-4, rtol=1e-4: {is_close_1e4}")
        logger.info(f"Close within atol=1e-5, rtol=1e-5: {is_close_1e5}")
        logger.info(f"Close within atol=1e-6, rtol=1e-6: {is_close_1e6}")
        logger.info("-" * 40)
        logger.info(f"PyTorch output range: [{np.min(pytorch_np):.6f}, {np.max(pytorch_np):.6f}]")
        logger.info(f"ExecuTorch output range: [{np.min(executorch_np):.6f}, {np.max(executorch_np):.6f}]")
        logger.info("=" * 60)

        # Save comparison results
        results_file = os.path.join(self.test_data_dir, "comparison_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Max Absolute Tolerance: {max_atol:.10f}\n")
            f.write(f"Mean Absolute Tolerance: {mean_atol:.10f}\n")
            f.write(f"Max Relative Tolerance: {max_rtol:.10f}\n")
            f.write(f"Mean Relative Tolerance: {mean_rtol:.10f}\n")
            f.write(f"Close within 1e-4: {is_close_1e4}\n")
            f.write(f"Close within 1e-5: {is_close_1e5}\n")
            f.write(f"Close within 1e-6: {is_close_1e6}\n")

        logger.info(f"Saved comparison results to: {results_file}")

        return max_atol, max_rtol, is_close_1e5

    def run_full_test(self):
        """Run the complete test pipeline."""
        logger.info("Starting Voxtral audio_encoder test pipeline...")

        try:
            # Setup
            self.setup_model_and_data()

            # Generate test inputs
            inputs = self.generate_test_inputs()

            # Export model
            pte_path, data_dir = self.export_audio_encoder()

            # Run PyTorch reference
            pytorch_output = self.run_pytorch_reference(inputs)

            # Run ExecuTorch inference
            executorch_output = self.run_executorch_inference(inputs)

            # Compare outputs
            max_atol, max_rtol, is_close = self.compare_outputs(pytorch_output, executorch_output)

            if is_close:
                logger.info("✅ TEST PASSED: Outputs are within acceptable tolerance")
                return True
            else:
                logger.warning("⚠️  TEST WARNING: Outputs exceed typical tolerance thresholds")
                logger.info("This may indicate numerical precision differences between implementations")
                return False

        except Exception as e:
            logger.error(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
            logger.info(f"Cleaned up test data directory: {self.test_data_dir}")


def main():
    # Print immediate feedback to show script is running
    print("🚀 Starting Voxtral audio_encoder test script...")
    print(f"Script location: {__file__}")

    parser = argparse.ArgumentParser(
        description="Test Voxtral audio_encoder with ExecuTorch Metal backend",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["export", "test", "compare"],
        default="compare",
        help="Test mode: export (export only), test (inference only), compare (full pipeline)"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help=f"Model ID to test (default: {MODEL_ID})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to use (default: {DEVICE})"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test data directory after completion"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    print(f"📋 Parsed arguments:")
    print(f"   Mode: {args.mode}")
    print(f"   Model ID: {args.model_id}")
    print(f"   Device: {args.device}")
    print(f"   Verbose: {args.verbose}")
    print(f"   Cleanup: {args.cleanup}")

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Verbose logging enabled")

    # Check if running on macOS for Metal support
    if args.device == "mps" and not torch.backends.mps.is_available():
        logger.error("Metal Performance Shaders (MPS) not available on this system")
        sys.exit(1)

    # Create tester instance
    tester = VoxtralAudioEncoderTester(args.model_id, args.device)

    success = False

    if args.mode == "export":
        tester.setup_model_and_data()
        tester.export_audio_encoder()
        success = True

    elif args.mode == "test":
        tester.setup_model_and_data()
        inputs = tester.generate_test_inputs()
        pytorch_output = tester.run_pytorch_reference(inputs)
        executorch_output = tester.run_executorch_inference()
        _, _, success = tester.compare_outputs(pytorch_output, executorch_output)

    elif args.mode == "executorch-inference":
        executorch_output = tester.run_executorch_inference()

    elif args.mode == "compare":
        success = tester.run_full_test()

    if success:
        logger.info("🎉 All tests completed successfully!")
    else:
        logger.warning("⚠️  Tests completed with warnings")


if __name__ == "__main__":
    main()
