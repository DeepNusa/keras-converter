from tensorflow.python.compiler.tensorrt import trt_convert as trt

def convert(savedmodel_path, use_fp16=False):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    if use_fp16:
        conversion_params = conversion_params._replace(precision_mode="FP16")
    
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=savedmodel_path,
        conversion_params=conversion_params
        )
    converter.convert()

    return converter