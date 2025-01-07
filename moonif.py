# --- moonshine --- #

import os
import time
from queue import Queue

import numpy as np
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer


_debug = True
caption_cache = None
transcribe = None

SAMPLING_RATE = 16000

CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MAX_LINE_LENGTH = 80

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2

MODEL_DIR = "/books/MachineLearning/Voice/misc/models/base"  # or None

# see "Moonshine Speech Recognition For Live Transcription and Voice Commands.pdf"
#
# The Moonshine models (https://huggingface.co/UsefulSensors/moonshine) are trained
# for the speech recognition task, capable of transcribing English speech audio into
# English text. Useful Sensors developed these models to support real time speech
# transcription on low cost hardware. There are 2 models of different size/capability:
# the tiny model has 27M parameters and the base model has 61M parameters.  Both
# support English only.
#
# These models are off-the-shelf encoder-decoder Transformer model (see 'Attention
# is all you need' paper) . The input is an audio signal sampled at 16,000 Hz. It does
# not use any hand-engineering to extract the audio features. Instead, the input is
# processed by a short stem of 3 convolution layers with strides 64, 3, and 2. These
# set of strides compress the input by a factor of 384x.  The model uses Rotary
# Position Embeddings (RoPE) at each layer of the encoder and decoder (see the
# 'Roformer: Enhanced transformer with rotary position embedding' paper) .
#
# The models were trained on 200,000 hours of audio and the corresponding transcripts
# collected from the internet, as well as datasets openly available and accessible on
# HuggingFace. The open datasets used are listed in the above linked paper.
#
# This ASR uses the same byte-level BPE text tokenizer as used in Llama 1 and 2 for
# tokenizing English text (see the 'Open foundation and fine-tuned chat Models' paper).
# The original vocabulary size was 32000; 768 special tokens were added.

class Transcriber(object):
    def __init__(self, model_name, rate=16000, model_dir=MODEL_DIR):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        # uses saved model if available
        self.model = MoonshineOnnxModel(models_dir=model_dir, model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, do_print=True, cbfunc=None):
    global transcribe
    global caption_cache

    """Transcribes, prints and caches the caption then clears speech buffer."""
    text = transcribe(speech)
    if do_print:
        print_captions(text)
    caption_cache.append(text)
    speech *= 0.0

    """Call the command processor with input text"""
    if cbfunc is not None:
        cbfunc(text)


def print_captions(text):
    global caption_cache

    """Prints right justified on same line, prepending cached captions."""
    if len(text) < MAX_LINE_LENGTH:
        for caption in caption_cache[::-1]:
            text = caption + " " + text
            if len(text) > MAX_LINE_LENGTH:
                break
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    else:
        text = " " * (MAX_LINE_LENGTH - len(text)) + text
    if _debug: print("moonshine: \r" + (" " * MAX_LINE_LENGTH) + "\r" + text, end="", flush=True)




def soft_reset(vad_iterator):
    """Soft resets Silero VADIterator without affecting VAD model state."""
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0



# consider use of "moonshine/tiny".  Also check onnx quantize reduction uses
# warden's performance optimize.

def moonshine_main_live_audio(cbfunc=None, model_name="moonshine/base"):
    global transcribe
    global caption_cache

    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    # see site-packages/silero_vad/model.py
    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    # stream for a PortAudio input stream (using NumPy)
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    with stream:
        print_captions("Ready...\n")

        try:
            while True:
                chunk, status = q.get()
                if status:
                    print(status)

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()
                        print("moonshine: start")

                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech, do_print=True, cbfunc=cbfunc)
                        print("moonshine: end")

                elif recording:
                    # Possible speech truncation can cause hallucination.

                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech)
                        soft_reset(vad_iterator)

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        print_captions(transcribe(speech))
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()

            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, do_print=False)

            print(f"""

             model_name :  {model_name}
       MIN_REFRESH_SECS :  {MIN_REFRESH_SECS}s

      number inferences :  {transcribe.number_inferences}
    mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
  model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
""")
            if caption_cache:
                print(f"Cached captions.\n{' '.join(caption_cache)}")



if __name__ == "__main__":
    moonshine_main_live_audio()