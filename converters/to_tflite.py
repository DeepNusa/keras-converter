import tensorflow as tf

def convert(model, use_fp16=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    return tflite_model

def save(model, output_path):
    with open(output_path, 'wb') as writer:
        writer.write(model)

