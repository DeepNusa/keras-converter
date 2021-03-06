# keras-converter

Convert your keras model into TF SavedModel, TFLite or Apple CoreML simply using one command

## How to?
- install tensorflow
- *(opt)* intall coremltools (if you need to convert your keras model into Apple CoreML, otherwise it's ok to skip)
- run the script by typing ```python3 convert.py -m /path/to/keras_model.h5 [target]```
  - **target** can be `--to-savedmodel`, `--to-tflite`, or `--to-coreml`
  - it can targeted more than one type at a time
  - if you want to convert to all available types, just use `--all`
  - for **tflite** and **coreml** model, you can add `--fp16` to use 16bit quantization instead of default 32bit
