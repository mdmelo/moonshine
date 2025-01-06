from pathlib import Path
import tokenizers
import keras
from .model import load_model, Moonshine

from . import ASSETS_DIR


# (env_moonshine) mike@t430sDebianBackup:/books/MachineLearning/SpeechToText$ python
# Python 3.11.5 (main, Sep 23 2023, 15:54:04) [GCC 10.2.1 20210110] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> import moonshine
# >>> moonshine.transcribe(moonshine.ASSETS_DIR / 'beckett.wav', 'moonshine/tiny')
# 2024-10-21 13:24:33.766457: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 37748736 exceeds 10% of free system memory.
# 2024-10-21 13:24:33.801059: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 37748736 exceeds 10% of free system memory.
# 2024-10-21 13:24:33.812808: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 37748736 exceeds 10% of free system memory.
# preprocessor.weights.h5: 100%|| 6.82M/6.82M [00:00<00:00, 31.6MB/s]
# encoder.weights.h5: 100%|| 24.2M/24.2M [00:00<00:00, 35.8MB/s]
# decoder.weights.h5: 100%|| 78.0M/78.0M [00:00<00:00, 110MB/s]
# 2024-10-21 13:24:37.550672: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 37748736 exceeds 10% of free system memory.
# /books/MachineLearning/SpeechToText/env_moonshine/lib/python3.11/site-packages/keras/src/ops/nn.py:545: UserWarning: You are using a softmax over axis 3 of a tensor of shape (1, 8, 1, 1). This axis has size 1. The softmax operation will always return the value 1, which is likely not what you intended. Did you mean to use a sigmoid instead?
#   warnings.warn(
# 2024-10-21 13:24:53.540012: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 37748736 exceeds 10% of free system memory.
# ['Ever tried ever failed, no matter try again, fail again, fail better.']



def load_audio(audio, return_numpy=False):
    if isinstance(audio, (str, Path)):
        import librosa

        audio, _ = librosa.load(audio, sr=16_000)
        if return_numpy:
            return audio[None, ...]
        audio = keras.ops.expand_dims(keras.ops.convert_to_tensor(audio), 0)
    return audio


def assert_audio_size(audio):
    assert len(keras.ops.shape(audio)) == 2, "audio should be of shape [batch, samples]"
    num_seconds = keras.ops.convert_to_numpy(keras.ops.size(audio) / 16_000)
    assert (
        0.1 < num_seconds < 64
    ), "Moonshine models support audio segments that are between 0.1s and 64s in a single transcribe call. For transcribing longer segments, pre-segment your audio and provide shorter segments."
    return num_seconds


def transcribe(audio, model="moonshine/base"):
    if isinstance(model, str):
        model = load_model(model)
    assert isinstance(
        model, Moonshine
    ), f"Expected a Moonshine model or a model name, not a {type(model)}"

    audio = load_audio(audio)
    assert_audio_size(audio)

    tokens = model.generate(audio)
    return load_tokenizer().decode_batch(tokens)


def load_tokenizer():
    tokenizer_file = ASSETS_DIR / "tokenizer.json"
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
    return tokenizer


def benchmark(audio, model="moonshine/base"):
    import time

    if isinstance(model, str):
        model = load_model(model)
    assert isinstance(
        model, Moonshine
    ), f"Expected a Moonshine model or a model name, not a {type(model)}"

    audio = load_audio(audio)
    num_seconds = assert_audio_size(audio)

    print("Warming up...")
    for _ in range(4):
        _ = model.generate(audio)

    print("Benchmarking...")
    N = 8
    start_time = time.time_ns()
    for _ in range(N):
        _ = model.generate(audio)
    end_time = time.time_ns()

    elapsed_time = (end_time - start_time) / N
    elapsed_time /= 1e6

    print(f"Time to transcribe {num_seconds:.2f}s of speech is {elapsed_time:.2f}ms")
