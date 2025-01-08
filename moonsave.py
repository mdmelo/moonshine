import sys
import keras

from moonshine import load_model, Moonshine
from pathlib import Path


def convert_and_store(model, input_signature, output_file):
    from tf2onnx.convert import from_keras
    import onnx

    onnx_model, external_storage_dict = from_keras(
        model, input_signature=input_signature
    )
    assert external_storage_dict is None, f"External storage for onnx not supported"
    onnx.save_model(onnx_model, output_file)


def store_model(model_name, outdir):
    # this is the default backend for keras v3.7.0
    assert (
        keras.config.backend() == "tensorflow"
    ), "Should be run with the tensorflow backend"

    import tensorflow as tf

    model = load_model(model_name)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    convert_and_store(
        model.preprocessor.preprocess,
        input_signature=[tf.TensorSpec([None, None], dtype=tf.float32)],
        output_file=f"{outdir}/preprocess.onnx",
    )

    seq_len_spec = tf.TensorSpec([1], dtype=tf.int32)

    convert_and_store(
        model.encoder.encoder,
        input_signature=[
            tf.TensorSpec([None, None, model.dim], dtype=tf.float32),
            seq_len_spec,
        ],
        output_file=f"{outdir}/encode.onnx",
    )

    input_spec = tf.TensorSpec([None, None], dtype=tf.int32)
    context_spec = tf.TensorSpec([None, None, model.dim], dtype=tf.float32)
    cache_spec = [
        tf.TensorSpec(
            [None, None, model.n_head, model.inner_dim // model.n_head],
            dtype=tf.float32,
        )
        for _ in range(model.dec_n_layers * 4)
    ]

    convert_and_store(
        model.decoder.uncached_call,
        input_signature=[input_spec, context_spec, seq_len_spec],
        output_file=f"{outdir}/uncached_decode.onnx",
    )

    convert_and_store(
        model.decoder.cached_call,
        input_signature=[input_spec, context_spec, seq_len_spec] + cache_spec,
        output_file=f"{outdir}/cached_decode.onnx",
    )



#     mike@t430sDebianBackup:/books/MachineLearning/Voice/misc/models$ ls -ls tiny/
#     116912 -rw-r--r-- 1 mike mike 119714037 Jan  6 12:55 cached_decode.onnx
#      29408 -rw-r--r-- 1 mike mike  30111620 Jan  6 12:52 encode.onnx
#       6644 -rw-r--r-- 1 mike mike   6800794 Jan  6 12:51 preprocess.onnx
#     124676 -rw-r--r-- 1 mike mike 127665192 Jan  6 12:53 uncached_decode.onnx
#
#     mike@t430sDebianBackup:/books/MachineLearning/Voice/misc/models$ ls -ls base/
#     226072 -rw-r--r-- 1 mike mike 231495454 Jan  6 12:44 cached_decode.onnx
#      81480 -rw-r--r-- 1 mike mike  83432884 Jan  6 12:39 encode.onnx
#      13748 -rw-r--r-- 1 mike mike  14077324 Jan  6 12:38 preprocess.onnx
#     247692 -rw-r--r-- 1 mike mike 253632530 Jan  6 12:42 uncached_decode.onnx


if __name__ == "__main__":
    # tiny -or- base
    model = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    model_name = "moonshine/" + model
    outdir = "./models/" + model

    store_model(model_name, outdir)
    print("saved {} model to {}".format(model_name, outdir))
