cmake --build cmake-out -j16 --target install --config Release
cmake -DEXECUTORCH_BUILD_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -Sexamples/models/voxtral \
      -Bcmake-out/examples/models/voxtral/
cmake --build cmake-out/examples/models/voxtral --target voxtral_runner --config Release
./cmake-out/examples/models/voxtral/voxtral_runner \
      --model_path $VOXTRAL_DIR/model.pte \
      --data_path $VOXTRAL_DIR/aoti_metal_blob.ptd \
      --tokenizer_path $VOXTRAL_DIR/tekken.json \
      --audio_path $VOXTRAL_DIR/call_samantha_hall.wav \
      --processor_path $VOXTRAL_DIR/voxtral_preprocessor.pte \
      --temperature 0 \
      --max_new_tokens 10 \
      --prompt "Translate this audio into Spanish"
