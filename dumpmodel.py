import onnx



# for model visualization:
#
# $ python venv/lib/python3.9/site-packages/onnx/tools/net_drawer.py --input preprocess.onnx --output ./preprocess.dot --embed_docstring
#
#
# models/base$ ls -lt
# total 569076
# -rw-r--r-- 1 mike mike     70784 Jan  8 14:05 preprocess.dot
# -rw-r--r-- 1 mike mike 253632530 Jan  6 14:40 uncached_decode.onnx
# -rw-r--r-- 1 mike mike  14077324 Jan  6 14:40 preprocess.onnx
# -rw-r--r-- 1 mike mike  83432884 Jan  6 14:40 encode.onnx
# -rw-r--r-- 1 mike mike 231495454 Jan  6 14:40 cached_decode.onnx
#
# $ dot -Tsvg preprocess.dot -o preprocess.svg
#
#
# see https://github.com/lutzroeder/netron
# see https://datascience.stackexchange.com/questions/12851/how-do-you-visualize-neural-network-architectures/19039
#
# Netron is a viewer for neural network, deep learning and machine learning models.
# Netron supports ONNX, TensorFlow Lite, Core ML, Keras, Caffe, Darknet, PyTorch, TensorFlow.js, Safetensors and NumPy.
# Netron has experimental support for TorchScript, TensorFlow, MXNet, OpenVINO, RKNN, ML.NET, ncnn, MNN, PaddlePaddle, GGUF and scikit-learn.
#
#
# $ uv pip install netron
# ...
# Installed 1 package in 32ms
#     + netron==8.0.9
#
# $ ls -l
# total 569268
# -rw-r--r-- 1 mike mike 231495454 Jan  6 14:40 cached_decode.onnx
# -rw-r--r-- 1 mike mike  83432884 Jan  6 14:40 encode.onnx
# -rw-r--r-- 1 mike mike  14077324 Jan  6 14:40 preprocess.onnx
# -rw-r--r-- 1 mike mike 253632530 Jan  6 14:40 uncached_decode.onnx
#
# $ netron ./preprocess.onnx
# Serving './preprocess.onnx' at http://localhost:8080
# ...


models = ["/books/MachineLearning/Voice/misc/models/tiny/preprocess.onnx",
          "/books/MachineLearning/Voice/misc/models/tiny/encode.onnx",
          "/books/MachineLearning/Voice/misc/models/tiny/uncached_decode.onnx",
          "/books/MachineLearning/Voice/misc/models/tiny/cached_decode.onnx"]

for modelname in models:
    onnxmodel= onnx.load(modelname)
    print("\n\nmodel {}".format(modelname))

    for i in onnxmodel.graph.initializer:
        print("    ", i.name)

    inits = onnxmodel.graph.initializer
    onnx_weights = {}

    for i in inits:
        w = onnx.numpy_helper.to_array(i)
        onnx_weights[i.name] = w

    for k,v in onnx_weights.items():
        # print("> {}".format(k))
        # print("{}: {}".format(k, v))
        pass



