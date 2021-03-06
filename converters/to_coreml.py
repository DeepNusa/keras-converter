import coremltools as ct

def convert(model, use_fp16=False):
    mlmodel = ct.convert(model)

    if use_fp16:
        from coremltools.models.neural_network import quantization_utils

        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel

