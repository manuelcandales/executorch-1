import logging
import os

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from torch.export.pt2_archive._package import package_pt2, load_pt2

from optimum.executorch import ExecuTorchModelForMultiModalToText
from optimum.exporters.executorch.tasks.multimodal_text_to_text import load_multimodal_text_to_text_model

from executorch.backends.apple.metal.metal_backend import MetalBackend
from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
from executorch.exir import to_edge_transform_and_lower

logging.basicConfig(level=logging.DEBUG)


def load_and_export(model_id, device):
    module = load_multimodal_text_to_text_model(
        model_id,
        device=device,
        use_custom_sdpa=False,
        use_custom_kv_cache=False,
    )

    ep = module.export()
    return ep

def get_audio_encoder(ep):
    return ep["audio_encoder"]

def get_token_embedding(ep):
    return ep["token_embedding"]

def get_text_decoder(ep):
    return ep["text_decoder"]

def executorch_metal_lowering(ep, name):
    device = "mps"
    output_data_dir_name = f"aoti_{name}_{device}"
    output_data_dir = os.path.join(os.getcwd(), "aoti_metal_data", output_data_dir_name)
    # Create the output_data_dir directory if it doesn't exist
    os.makedirs(output_data_dir, exist_ok=True)

    output_pte_path = f"aoti_{name}_{device}.pte"

    aten_dialect = ep[name]

    # 2. to_edge: Make optimizations for Edge devices
    print("Step 3: Lowering to Edge dialect...")
    edge_program = to_edge_transform_and_lower(
        aten_dialect, partitioner=[MetalPartitioner([MetalBackend.generate_method_name_compile_spec(name)])]
    )
    print("Lowering to Edge dialect done.")

    # 3. to_executorch: Convert the graph to an ExecuTorch program
    print("Step 4: Converting to ExecuTorch program...")
    executorch_program = edge_program.to_executorch()
    print("To executorch done.")

    # 4. Save the compiled .pte program
    if output_data_dir is None:
        output_data_dir = os.getcwd()

    print(f"Step 5: Saving pte to {output_pte_path} and ptd to {output_data_dir}")
    with open(output_pte_path, "wb") as file:
        file.write(executorch_program.buffer)

    print(f"size of Named Data: {len(executorch_program._tensor_data)}")

    executorch_program.write_tensor_data_to_file(output_data_dir)

    print(
        f"Export completed successfully! PTE saved to {output_pte_path} and ptd saved to {output_data_dir}"
    )

def compile_ep(name, ep_, device):
    print(f"AOTI Compiling {name}")

    path = torch._inductor.aoti_compile_and_package(
        ep_,
        package_path=f"aoti_{name}_{device}.pt2",
    )

def export_model(model_id, device):
    ep = load_and_export(model_id, device)
    for name, ep_ in ep.items():
        compile_ep(name, ep_, device)

def main_executorch(device, mode, name):
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    config = AutoConfig.from_pretrained(model_id)

    if mode == "export":
        ep = load_and_export(model_id, device)
        executorch_metal_lowering(ep, name)
        return

def main_torch(device, mode):
    model_id = "mistralai/Voxtral-Mini-3B-2507"
    config = AutoConfig.from_pretrained(model_id)

    if mode == "export":
        export_model(model_id, device)
        return

    ep = {
        "token_embedding": torch._inductor.aoti_load_package("aoti_token_embedding_mps.pt2"),
        "audio_encoder": torch._inductor.aoti_load_package("aoti_audio_encoder_mps.pt2"),
        "text_decoder": torch._inductor.aoti_load_package("aoti_text_decoder_mps.pt2"),
    }

    # Generate
    print(f"\nGenerating...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor.apply_chat_template(conversation)

    for name in inputs:
        inputs[name] = inputs[name].to(device)

    input_ids = inputs["input_ids"]
    print("Producing token embeddings with exported program ...")
    token_embeddings = ep["token_embedding"](input_ids)

    if "input_features" in inputs:
        input_features = inputs["input_features"]
        print("Producing audio embeddings with exported program ...")

        audio_embeddings = ep["audio_encoder"](input_features=input_features)

    audio_token_mask = inputs["input_ids"] == config.audio_token_id
    token_embeddings[audio_token_mask] = audio_embeddings

    # Prefill prompt embeddings
    print("Producing text decoded with exported program ...")
    logits = ep["text_decoder"](inputs_embeds=token_embeddings, cache_position=torch.arange(token_embeddings.shape[1], dtype=torch.long, device=device))

    token = torch.argmax(logits[:, -1, :])

    tokens = [token.item()]
    print("Generated:", tokenizer.decode([token.item()]), end="\n")

    pos = token_embeddings.shape[1]

    max_generation_len = 64

    while pos < input_ids.shape[-1] + max_generation_len:
        token_embedding = ep["token_embedding"](token.unsqueeze(0).unsqueeze(0))
        logits = (
            ep["text_decoder"](
                inputs_embeds=token_embedding,
                cache_position=torch.tensor([pos], dtype=torch.long, device=device),
            )
        )
        token = torch.argmax(logits[:, -1, :])
        print("Generated:", tokenizer.decode([token.item()]), end="\n")
        tokens.append(token.item())
        pos += 1

    output = tokenizer.decode(tokens, skip_special_tokens=True)
    print("\nFull Output:")
    print(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Voxtral-Mini-3B-2507 inference with optional AOTI compilation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["export", "run"],
        help="`export` exports the model to a .pt2, `run` loads and runs the pt2",
    )
    parser.add_argument(
        "--name",
        type=str,
        choices=["token_embedding", "audio_encoder", "text_decoder"],
        help="Component to export and run",
    )
    args = parser.parse_args()

    main_executorch(args.device, args.mode, args.name)
