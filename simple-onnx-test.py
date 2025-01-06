# This script demonstrates how to run a Moonshine model with the `onnxruntime` package alone, 
# without depending on `torch` or `tensorflow`. This enables running on SBCs such as Raspberry Pi. 
# Follow the instructions below to setup and run.
#
# * Install `onnxruntime` (or `onnxruntime-gpu` if you want to run on GPUs) and `tokenizers` 
# packages using your Python package manager of choice, such as `pip`.
#
# * Download the `onnx` files from huggingface hub to a directory.
#     ```shell
#     mkdir moonshine_base_onnx
#     cd moonshine_base_onnx
#     wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/preprocess.onnx
#     wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/encode.onnx
#     wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/uncached_decode.onnx
#     wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/base/cached_decode.onnx
#     cd ..
#     ```
# * Run `onnx_standalone.py` to transcribe a wav file
#     ```shell
#     moonshine/moonshine/demo/onnx_standalone.py --models_dir moonshine_base_onnx --wav_file moonshine/moonshine/assets/beckett.wav
#     ['Ever tried ever failed, no matter try again fail again fail better.']
#     ```
    
import os
import sys
import argparse
import wave
import numpy as np
import tokenizers

MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))
from onnx_model import MoonshineOnnxModel

def main(models_dir, wav_file):
    m = MoonshineOnnxModel(models_dir=models_dir)
    with wave.open(wav_file) as f:
        params = f.getparams()
        assert (
            params.nchannels == 1
            and params.framerate == 16_000
            and params.sampwidth == 2
        ), f"wave file should have 1 channel, 16KHz, and int16"
        audio = f.readframes(params.nframes)
    audio = np.frombuffer(audio, np.int16) / 32768.0
    audio = audio.astype(np.float32)[None, ...]
    tokens = m.generate(audio)
    tokenizer = tokenizers.Tokenizer.from_file(
        os.path.join(MOONSHINE_DEMO_DIR, "..", "assets", "tokenizer.json")
    )
    text = tokenizer.decode_batch(tokens)
    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="onnx_standalone",
        description="Standalone ONNX demo of Moonshine models",
    )
    parser.add_argument(
        "--models_dir", help="Directory containing ONNX files", required=True
    )
    parser.add_argument("--wav_file", help="Speech WAV file", required=True)
    args = parser.parse_args()
    main(**vars(args))