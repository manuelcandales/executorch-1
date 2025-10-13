#!/bin/bash
# Combined ExecuTorch Voxtral Export and E2E Test Script for Metal Backend
# Converted from CUDA workflow steps: export-voxtral-cuda-artifact and test-voxtral-cuda-e2e
#
# Changes made from original CUDA steps:
# 1. Replaced CMAKE_ARGS="-DEXECUTORCH_BUILD_CUDA=ON" with CMAKE_ARGS="-DEXECUTORCH_BUILD_METAL=ON"
# 2. Changed optimum-cli export --recipe from "cuda" to "metal" and --device from "cuda" to "metal"
# 3. Updated artifact filenames from aoti_cuda_blob.ptd to aoti_metal_blob.ptd
# 4. Updated all cmake build flags from -DEXECUTORCH_BUILD_CUDA=ON to -DEXECUTORCH_BUILD_METAL=ON
# 5. Changed runner target from linux.g5.4xlarge.nvidia.gpu to macOS (Metal requires Apple hardware)
# 6. Updated artifact and job names from "cuda" to "metal"
# 7. Removed CUDA-specific LD_LIBRARY_PATH exports (not needed for Metal on macOS)
# 8. Added proper LLM preset and build sequence to enable EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER

set -eux

# ========================
# EXPORT VOXTRAL METAL ARTIFACT (formerly export-voxtral-cuda-artifact)
# ========================

echo "::group::Setup ExecuTorch"
CMAKE_ARGS="-DEXECUTORCH_BUILD_METAL=ON" ./install_executorch.sh
echo "::endgroup::"

echo "::group::Setup Huggingface"
pip install -U "huggingface_hub[cli]" accelerate
huggingface-cli login --token $SECRET_EXECUTORCH_HF_TOKEN
OPTIMUM_ET_VERSION=$(cat .ci/docker/ci_commit_pins/optimum-executorch.txt)
pip install git+https://github.com/huggingface/optimum-executorch.git@${OPTIMUM_ET_VERSION}
pip install mistral-common librosa
pip list
echo "::endgroup::"

echo "::group::Export Voxtral"
optimum-cli export executorch \
    --model "mistralai/Voxtral-Mini-3B-2507" \
    --task "multimodal-text-to-text" \
    --recipe "metal" \
    --dtype bfloat16 \
    --max_seq_len 1024 \
    --output_dir ./
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 128 \
    --stack_output \
    --max_audio_len 300 \
    --output_file voxtral_preprocessor.pte

test -f model.pte
test -f aoti_metal_blob.ptd
test -f voxtral_preprocessor.pte
echo "::endgroup::"

echo "::group::Store Voxtral Artifacts"
mkdir -p voxtral_metal_artifacts
cp model.pte voxtral_metal_artifacts/
cp aoti_metal_blob.ptd voxtral_metal_artifacts/
cp voxtral_preprocessor.pte voxtral_metal_artifacts/
ls -al voxtral_metal_artifacts/
echo "::endgroup::"

# ========================
# TEST VOXTRAL METAL E2E (formerly test-voxtral-cuda-e2e)
# ========================

echo "::group::Setup ExecuTorch Requirements"
CMAKE_ARGS="-DEXECUTORCH_BUILD_METAL=ON" ./install_requirements.sh
pip list
echo "::endgroup::"

echo "::group::Prepare Voxtral Artifacts"
cp "voxtral_metal_artifacts/model.pte" .
cp "voxtral_metal_artifacts/aoti_metal_blob.ptd" .
cp "voxtral_metal_artifacts/voxtral_preprocessor.pte" .
TOKENIZER_URL="https://huggingface.co/mistralai/Voxtral-Mini-3B-2507/resolve/main/tekken.json"
curl -L $TOKENIZER_URL -o tekken.json
ls -al model.pte aoti_metal_blob.ptd voxtral_preprocessor.pte tekken.json
echo "::endgroup::"

echo "::group::Download Test Audio File"
AUDIO_URL="https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/testaudio/16000/test01_20s.wav"
curl -L $AUDIO_URL -o poem.wav
echo "::endgroup::"



# echo "::group::Build Voxtral Benchmark"
# cmake -DCMAKE_BUILD_TYPE=Release \
#       -DEXECUTORCH_BUILD_METAL=ON \
#       -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
#       -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
#       -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
#       -Bcmake-out .
# cmake --build cmake-out -j6 --target voxtral_runner
# echo "::endgroup::"

# echo "::group::Run Voxtral Benchmark"

# export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
# cmake-out/backends/cuda/voxtral_runner model.pte aoti_cuda_blob.ptd


CMAKE_ARGS="$CMAKE_ARGS -DEXECUTORCH_LOG_LEVEL=Debug -DCMAKE_BUILD_TYPE=Debug"

echo "::group::Build Voxtral Runner"
cmake --preset llm \
      -DEXECUTORCH_BUILD_METAL=ON \
      -DCMAKE_INSTALL_PREFIX=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_ENABLE_LOGGING=OFF \
      -DEXECUTORCH_LOG_LEVEL=Debug \
      -Bcmake-out -S.
cmake --build cmake-out -j16 --target install --config Release
# cmake --build cmake-out -j$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu) - 1 )) --target install --config Release

cmake -DEXECUTORCH_BUILD_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -Sexamples/models/voxtral \
      -Bcmake-out/examples/models/voxtral/
cmake --build cmake-out/examples/models/voxtral --target voxtral_runner --config Release
echo "::endgroup::"

echo "::group::Run Voxtral Runner"
set +e
# Note: LD_LIBRARY_PATH export removed as it's Linux/CUDA specific and not needed on macOS Metal
OUTPUT=$(cmake-out/examples/models/voxtral/voxtral_runner \
      --model_path model.pte \
      --data_path aoti_metal_blob.ptd \
      --tokenizer_path tekken.json \
      --audio_path poem.wav \
      --processor_path voxtral_preprocessor.pte \
      --temperature 0 2>&1)
EXIT_CODE=$?
set -e

echo "$OUTPUT"

if ! echo "$OUTPUT" | grep -iq "poem"; then
  echo "Expected output 'poem' not found in output"
  exit 1
fi

if [ $EXIT_CODE -ne 0 ]; then
  echo "Unexpected exit code: $EXIT_CODE"
  exit $EXIT_CODE
fi
echo "::endgroup::"

echo "SUCCESS: Voxtral Metal export and e2e test completed successfully!"




export VOXTRAL_DIR=/Users/mcandales/github/metal_delegate/voxtral

./cmake-out/examples/models/voxtral/voxtral_runner \
      --model_path $VOXTRAL_DIR/model.pte \
      --data_path $VOXTRAL_DIR/aoti_metal_blob.ptd \
      --tokenizer_path $VOXTRAL_DIR/tekken.json \
      --audio_path $VOXTRAL_DIR/poem.wav \
      --processor_path $VOXTRAL_DIR/voxtral_preprocessor.pte \
      --temperature 0
