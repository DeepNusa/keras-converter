import os
import tensorflow as tf

class MasterConverter:
    def __init__(self, args):
        self.args = args

        model_path = self.args.model
        self.model = tf.keras.models.load_model(model_path)

        self.root, model_name = os.path.split(model_path)
        self.model_name = os.path.splitext(model_name)[0]
    
    def process(self):
        if self.args.to_savedmodel or self.args.to_tftrt or self.args.all:
            self.model.save(os.path.join(self.root, self.model_name))

        if self.args.to_tflite or self.args.all:
            from .to_tflite import convert, save

            tflite_model = convert(self.model, use_fp16=self.args.fp16)
            save(tflite_model, os.path.join(self.root, self.model_name+'.tflite'))
        
        if self.args.to_coreml or self.args.all:
            from .to_coreml import convert

            mlmodel = convert(self.model, use_fp16=self.args.fp16)
            mlmodel.save(os.path.join(self.root, self.model_name+'.mlmodel'))
        
        if self.args.to_tftrt or self.args.all:
            from .to_tftrt import convert

            tftrt_model = convert(os.path.join(self.root, self.model_name), use_fp16=self.args.fp16)
            tftrt_model.save(os.path.join(self.root, self.model_name+'_trt'))
